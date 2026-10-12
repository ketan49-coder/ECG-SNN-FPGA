import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Checkpoint Selection on Validation Macro-F1 (4-Class AAMI)
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FEATURES   = 128
HIDDEN_1_NEURONS = 32
HIDDEN_2_NEURONS = 32
OUTPUT_CLASSES   = 4          # 4-Class: N, S, V, F
NUM_STEPS        = 25
BETA             = 0.9375     # 1 - 2^(-4) (Hardware bit-shift leak)
THRESHOLD        = 0.5        # Q8.8 = x"0080"
BATCH_SIZE       = 128
NUM_EPOCHS       = 60
LEARNING_RATE    = 2e-3
DROPOUT_P        = 0.3
DATA_DIR         = "data_ds"

if os.path.exists("/content/drive/MyDrive/SNN_Models"):
    MODEL_DIR = "/content/drive/MyDrive/SNN_Models"
else:
    MODEL_DIR = "models"

CLASS_NAMES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VanRossumLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.decay = 0.9375
        self.num_classes = num_classes

    def forward(self, spk_out, targets):
        T, B, C = spk_out.shape
        target_spikes = torch.zeros(T, B, C, device=spk_out.device)
        for b in range(B):
            target_spikes[:, b, targets[b]] = 1.0

        filtered_out = torch.zeros_like(spk_out)
        filtered_tgt = torch.zeros_like(target_spikes)
        
        trace_out = torch.zeros(B, C, device=spk_out.device)
        trace_tgt = torch.zeros(B, C, device=spk_out.device)

        for t in range(T):
            trace_out = trace_out * self.decay + spk_out[t]
            trace_tgt = trace_tgt * self.decay + target_spikes[t]
            filtered_out[t] = trace_out
            filtered_tgt[t] = trace_tgt

        loss = F.mse_loss(filtered_out, filtered_tgt, reduction='none')
        return loss.sum(dim=(0, 2)).mean()

class ECG_SuperSpike(nn.Module):
    def __init__(self, dropout_p=DROPOUT_P):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(INPUT_FEATURES, HIDDEN_1_NEURONS, bias=False)
        self.fc2 = nn.Linear(HIDDEN_1_NEURONS, HIDDEN_2_NEURONS, bias=False)
        self.fc3 = nn.Linear(HIDDEN_2_NEURONS, OUTPUT_CLASSES, bias=False)

        self.lif1 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")
        self.lif2 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")
        self.lif3 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")
        self.drop = nn.Dropout(p=dropout_p)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []
        drop_mask = torch.rand(NUM_STEPS) > 0.1 if self.training else torch.ones(NUM_STEPS, dtype=torch.bool)

        for t in range(NUM_STEPS):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.drop(spk1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            if not drop_mask[t]:
                spk3_rec.append(torch.zeros(x.size(0), OUTPUT_CLASSES, device=x.device))
            else:
                spk3_rec.append(spk3)

        return torch.stack(spk3_rec)

def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val,   dtype=torch.float32), torch.tensor(y_val,   dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test,  dtype=torch.long))
    
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
            DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0))

@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    all_preds, all_targets = [], []
    for data, targets in loader:
        data = data.to(device)
        spk_out = net(data)
        preds   = spk_out.sum(0).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.numpy())
        
    all_targets = np.array(all_targets)
    all_preds   = np.array(all_preds)
    acc = 100.0 * np.mean(all_preds == all_targets)
    macro_f1 = 100.0 * f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, macro_f1, all_targets, all_preds

def train_model():
    train_loader, val_loader, test_loader = load_data()
    net = ECG_SuperSpike().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn = VanRossumLoss(num_classes=OUTPUT_CLASSES)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    best_val_f1 = 0.0
    best_epoch = 0
    best_model_path = os.path.join(MODEL_DIR, "snn_superspike_ds1_ds2_best.pth")

    print("\n" + "=" * 80)
    print("STEP 1: DS1/DS2 RETRAIN WITH VALIDATION MACRO-F1 CHECKPOINT SELECTION")
    print("=" * 80)
    print("Validation Set: 7 Unseen DS1 Patients (stable, low-noise representation)")
    print("Checkpoint Rule: Best VALIDATION MACRO-F1 (penalizes majority-class bias)")
    print("Test Set: DS2 (22 completely unseen patients, evaluated ONCE at the end)")
    print("-" * 80)
    print(f"{'Ep':>4} | {'Loss':>8} | {'Train%':>7} | {'Val Acc%':>9} | {'Val Macro-F1%':>14} | {'LR':>8}")
    print("-" * 80)

    for epoch in range(NUM_EPOCHS):
        net.train()
        sum_loss = 0.0
        correct, total = 0, 0
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            spk_out = net(data)
            loss = loss_fn(spk_out, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            sum_loss += loss.item()
            preds = spk_out.sum(0).argmax(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_acc = 100.0 * correct / total
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Evaluate on the 7 Validation Patients
        val_acc, val_f1, _, _ = evaluate(net, val_loader)
        n = len(train_loader)

        marker = ""
        # CHECKPOINT SELECTION ON VALIDATION MACRO-F1!
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(net.state_dict(), best_model_path)
            marker = " ★ BEST"

        print(f"{epoch+1:>4} | {sum_loss/n:>8.4f} | {train_acc:>6.2f}% | {val_acc:>8.2f}% | {val_f1:>13.2f}% | {current_lr:.2e}{marker}")

    print("=" * 80)
    print(f"Training Complete! Best Checkpoint was Epoch {best_epoch} (Val Macro-F1: {best_val_f1:.2f}%)")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL TEST EVALUATION ON UNSEEN DS2 PATIENTS
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint from Epoch {best_epoch} for DS2 test evaluation...")
    net.load_state_dict(torch.load(best_model_path, map_location=device))
    test_acc, test_f1, test_targets, test_preds = evaluate(net, test_loader)

    print("\n" + "=" * 80)
    print("FINAL INDEPENDENT EVALUATION ON DS2 (22 UNSEEN TEST PATIENTS):")
    print("=" * 80)
    print(f"DS2 Overall Test Accuracy : {test_acc:.2f}%")
    print(f"DS2 Test Macro-F1 Score   : {test_f1:.2f}%\n")
    print(classification_report(test_targets, test_preds, target_names=CLASS_NAMES, digits=4))
    print("Confusion Matrix on DS2:")
    print(confusion_matrix(test_targets, test_preds))
    print("=" * 80)

if __name__ == "__main__":
    train_model()
