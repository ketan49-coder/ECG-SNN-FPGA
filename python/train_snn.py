import os
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import spikegen
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Network Architecture and Parameters
INPUT_FEATURES = 128
HIDDEN_1_NEURONS = 16
HIDDEN_2_NEURONS = 16
OUTPUT_CLASSES = 5
NUM_STEPS = 16      # Timesteps for SNN simulation
BETA = 0.9375       # Decay rate (1 - 1/16, implemented as bit-shift on FPGA)
THRESHOLD = 0.5     # Lowered from 1.0 to fix vanishing spike problem in deep SNNs
BATCH_SIZE = 128
NUM_EPOCHS = 50
LEARNING_RATE = 5e-3  # Increased from 1e-3 to help overcome flat gradients
DATA_DIR = "data"

# Setup device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

def compute_class_weights(y_train):
    """
    Compute inverse-frequency class weights to handle severe class imbalance.
    The MIT-BIH dataset is ~71% Normal, so we penalize the model much more
    heavily when it misses a rare Ventricular or Fusion beat.
    """
    unique, counts = np.unique(y_train, return_counts=True)
    class_names = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']
    
    print("\nClass distribution in training set:")
    for c, count in zip(unique, counts):
        print(f"  {class_names[c]}: {count} samples")
    
    # Inverse frequency weighting: rarer classes get higher weight
    total = len(y_train)
    weights = total / (len(unique) * counts)
    
    print("\nClass weights applied to loss function:")
    for c, w in zip(unique, weights):
        print(f"  {class_names[c]}: {w:.4f}")
        
    return torch.tensor(weights, dtype=torch.float32).to(device)

def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset  = TensorDataset(torch.tensor(X_test),  torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, y_train

# Define Network Architecture (128 -> 16 -> 16 -> 5)
class ECG_SNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Hidden Layer 1 (128 -> 16)
        self.fc1  = nn.Linear(INPUT_FEATURES,    HIDDEN_1_NEURONS, bias=False)
        self.lif1 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")
        
        # Hidden Layer 2 (16 -> 16)
        self.fc2  = nn.Linear(HIDDEN_1_NEURONS, HIDDEN_2_NEURONS, bias=False)
        self.lif2 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")
        
        # Output Layer (16 -> 5)
        self.fc3  = nn.Linear(HIDDEN_2_NEURONS, OUTPUT_CLASSES, bias=False)
        self.lif3 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")

    def forward(self, x):
        # Rate coding: convert normalized [0,1] amplitude to spike trains
        spike_data = spikegen.rate(x, num_steps=NUM_STEPS)
        
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        spk3_rec = []
        mem3_rec = []

        for step in range(NUM_STEPS):
            cur1 = self.fc1(spike_data[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec)

def train_model():
    train_loader, test_loader, y_train = load_data()
    
    # Compute class weights based on training distribution
    class_weights = compute_class_weights(y_train)
    
    net = ECG_SNN().to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    
    # CrossEntropy with class weights: rare classes get penalized much more
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    # Learning rate scheduler: reduce LR when accuracy stops improving
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    print(f"\nStarting Training: 128 -> 16 -> 16 -> 5 | threshold={THRESHOLD} | lr={LEARNING_RATE}")
    print("-" * 70)
    
    best_test_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        net.train()
        total_loss = 0
        correct    = 0
        total      = 0
        
        for data, targets in train_loader:
            data    = data.to(device)
            targets = targets.to(device)
            
            spk_out, mem_out = net(data)
            
            # Sum spikes over all timesteps → class scores [batch, 5]
            spike_sum = spk_out.sum(dim=0)
            
            # Weighted cross entropy loss
            loss = loss_fn(spike_sum, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = spike_sum.max(1)
            total   += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_acc = 100.0 * correct / total
        scheduler.step()
        
        # Evaluate on test set every 5 epochs
        if (epoch + 1) % 5 == 0:
            net.eval()
            t_correct = 0
            t_total   = 0
            with torch.no_grad():
                for data, targets in test_loader:
                    data    = data.to(device)
                    targets = targets.to(device)
                    spk_out, _ = net(data)
                    _, predicted = spk_out.sum(dim=0).max(1)
                    t_total   += targets.size(0)
                    t_correct += (predicted == targets).sum().item()
            test_acc = 100.0 * t_correct / t_total
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                os.makedirs('models', exist_ok=True)
                torch.save(net.state_dict(), "models/snn_best.pth")
                print(f"  --> New best model saved! Test Acc: {test_acc:.2f}%")
            
            print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")
        else:
            print(f"Epoch [{epoch+1:3d}/{NUM_EPOCHS}] | Loss: {total_loss/len(train_loader):.4f} | Train: {train_acc:.2f}%")

    print(f"\nBest Test Accuracy achieved: {best_test_acc:.2f}%")
    print("Best model saved to models/snn_best.pth")

if __name__ == "__main__":
    train_model()
