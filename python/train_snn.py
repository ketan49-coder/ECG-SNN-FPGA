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
NUM_STEPS = 16  # Timesteps for SNN simulation
BETA = 0.9375   # Decay rate (equivalent to bit-shifting by 4 in hardware)
BATCH_SIZE = 128
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
DATA_DIR = "data"

# Setup device (use GPU if available)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

# Define Network Architecture (128 -> 16 -> 16 -> 5)
class ECG_SNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Surrogate gradient for training spiking neurons
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Hidden Layer 1 (128 -> 16)
        self.fc1 = nn.Linear(INPUT_FEATURES, HIDDEN_1_NEURONS, bias=False)
        self.lif1 = snn.Leaky(beta=BETA, spike_grad=spike_grad, reset_mechanism="zero")
        
        # Hidden Layer 2 (16 -> 16)
        self.fc2 = nn.Linear(HIDDEN_1_NEURONS, HIDDEN_2_NEURONS, bias=False)
        self.lif2 = snn.Leaky(beta=BETA, spike_grad=spike_grad, reset_mechanism="zero")
        
        # Output Layer (16 -> 5)
        self.fc3 = nn.Linear(HIDDEN_2_NEURONS, OUTPUT_CLASSES, bias=False)
        self.lif3 = snn.Leaky(beta=BETA, spike_grad=spike_grad, reset_mechanism="zero")

    def forward(self, x):
        # Generate spike trains using rate coding based on the input amplitude
        spike_data = spikegen.rate(x, num_steps=NUM_STEPS)
        
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record output spikes and membrane potentials over time
        spk3_rec = []
        mem3_rec = []

        for step in range(NUM_STEPS):
            # Layer 1
            cur1 = self.fc1(spike_data[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # Layer 2
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Output Layer
            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec)

def train_model():
    train_loader, test_loader = load_data()
    net = ECG_SNN().to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    loss_fn = snn.functional.ce_rate_loss()

    print("Starting Training on 128->16->16->5 architecture...")
    
    for epoch in range(NUM_EPOCHS):
        net.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            spk_out, mem_out = net(data)
            loss = loss_fn(spk_out, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = spk_out.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_out, _ = net(data)
            _, predicted = spk_out.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    test_acc = 100.0 * correct / total
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    os.makedirs('models', exist_ok=True)
    torch.save(net.state_dict(), "models/snn_model.pth")
    print("Model saved to models/snn_model.pth")

if __name__ == "__main__":
    train_model()
