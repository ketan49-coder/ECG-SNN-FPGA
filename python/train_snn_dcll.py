import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────
INPUT_FEATURES   = 128
HIDDEN_1_NEURONS = 32
HIDDEN_2_NEURONS = 32
OUTPUT_CLASSES   = 5
NUM_STEPS        = 25
BETA             = 0.9375
THRESHOLD        = 0.5
BATCH_SIZE       = 128
NUM_EPOCHS       = 60
LEARNING_RATE    = 2e-3
DROPOUT_P        = 0.3
DATA_DIR         = "data"
MODEL_DIR        = "/content/drive/MyDrive/SNN_Models"
CLASS_NAMES      = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# van Rossum Distance Loss (Hardware Shift-Decay)
# ─────────────────────────────────────────────
class VanRossumLoss(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # HW CONSTRAINT 1: Shift-Decay Traces
        # Instead of np.exp(-1/tau), the FPGA uses trace - (trace >> 4)
        # This is exactly 15/16 = 0.9375. We enforce this in the software.
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

# ─────────────────────────────────────────────
# DCLL Network (Deep Continuous Local Learning)
# ─────────────────────────────────────────────
class DCLL_SuperSpike(nn.Module):
    def __init__(self, dropout_p=DROPOUT_P):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Standard Forward Weights (Trained)
        self.fc1 = nn.Linear(INPUT_FEATURES, HIDDEN_1_NEURONS, bias=False)
        self.fc2 = nn.Linear(HIDDEN_1_NEURONS, HIDDEN_2_NEURONS, bias=False)
        self.fc3 = nn.Linear(HIDDEN_2_NEURONS, OUTPUT_CLASSES, bias=False)

        # Force initial weights into strict Q1.7 quantization (multiples of 1/128)
        with torch.no_grad():
            for fc in [self.fc1, self.fc2, self.fc3]:
                fc.weight.data = torch.round(fc.weight.data * 128.0) / 128.0
                fc.weight.data.clamp_(-1.0, 127.0/128.0)

        # Fixed Random Projections for Local Cost (NOT Trained)
        self.R1 = nn.Parameter(torch.randn(HIDDEN_1_NEURONS, OUTPUT_CLASSES) / np.sqrt(HIDDEN_1_NEURONS), requires_grad=False)
        self.R2 = nn.Parameter(torch.randn(HIDDEN_2_NEURONS, OUTPUT_CLASSES) / np.sqrt(HIDDEN_2_NEURONS), requires_grad=False)

        self.lif1 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")
        self.lif2 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")
        self.lif3 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")
        self.drop = nn.Dropout(p=dropout_p)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        local_spk1_rec = []
        local_spk2_rec = []
        spk3_rec = []
        
        drop_mask = torch.rand(NUM_STEPS) > 0.1 if self.training else torch.ones(NUM_STEPS, dtype=torch.bool)

        for t in range(NUM_STEPS):
            # --- LAYER 1 ---
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_drop = self.drop(spk1)
            
            # Local Projection 1
            if not drop_mask[t]:
                local_spk1_rec.append(torch.zeros(x.size(0), OUTPUT_CLASSES, device=x.device))
            else:
                local_spk1_rec.append(spk1_drop @ self.R1)
            
            # --- LAYER 2 ---
            # DETACH: Stop gradients from flowing back to Layer 1
            cur2 = self.fc2(spk1_drop.detach())
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Local Projection 2
            if not drop_mask[t]:
                local_spk2_rec.append(torch.zeros(x.size(0), OUTPUT_CLASSES, device=x.device))
            else:
                local_spk2_rec.append(spk2 @ self.R2)
            
            # --- LAYER 3 ---
            # DETACH: Stop gradients from flowing back to Layer 2
            cur3 = self.fc3(spk2.detach())
            spk3, mem3 = self.lif3(cur3, mem3)
            
            if not drop_mask[t]:
                spk3_rec.append(torch.zeros(x.size(0), OUTPUT_CLASSES, device=x.device))
            else:
                spk3_rec.append(spk3)

        return torch.stack(local_spk1_rec), torch.stack(local_spk2_rec), torch.stack(spk3_rec)


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(X_test,  dtype=torch.float32), torch.tensor(y_test,  dtype=torch.long))
    return (DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0),
            DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0))

@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    correct = torch.zeros(OUTPUT_CLASSES)
    total   = torch.zeros(OUTPUT_CLASSES)
    t_ok, t_tot = 0, 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        _, _, final3 = net(data)
        preds = final3.sum(0).argmax(1)
        t_ok  += (preds == targets).sum().item()
        t_tot += targets.size(0)
        for c in range(OUTPUT_CLASSES):
            m = (targets == c)
            total[c]   += m.sum().item()
            correct[c] += (preds[m] == c).sum().item()
    overall = 100.0 * t_ok / t_tot
    per_cls = 100.0 * correct / total.clamp(min=1)
    return overall, per_cls

def train_model():
    train_loader, test_loader = load_data()
    net = DCLL_SuperSpike().to(device)
    
    # Note: We still use an optimizer object to hold state/learning rates, 
    # but we bypass optimizer.step() below to manually simulate hardware updates.
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    loss_fn = VanRossumLoss(num_classes=OUTPUT_CLASSES)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)
    os.makedirs(MODEL_DIR, exist_ok=True)
    best_acc = 0.0

    print(f"\nTraining DCLL (Local Errors) + HW Constraints (8-bit Stochastic Rounding)")
    print("=" * 70)
    print(f"{'Ep':>4} | {'TotalLoss':>9} | {'Train%':>7} | {'Test%':>7} | {'Gap':>6} | {'LR':>8}")
    print("-" * 70)

    # Q1.7 Hardware Step Size
    DELTA = 1.0 / 128.0

    for epoch in range(NUM_EPOCHS):
        net.train()
        sum_loss = 0.0
        correct, total = 0, 0
        current_lr = scheduler.get_last_lr()[0]
        
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass returns local projections and final output
            local1, local2, final3 = net(data)
            
            # Calculate Local Losses
            loss1 = loss_fn(local1, targets)
            loss2 = loss_fn(local2, targets)
            loss3 = loss_fn(final3, targets)
            
            total_loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            total_loss.backward()
            
            # ─────────────────────────────────────────────
            # HW CONSTRAINT: Read-Modify-Write w/ Stochastic Rounding
            # ─────────────────────────────────────────────
            # Bypassing optimizer.step() to simulate FPGA 8-bit updates
            with torch.no_grad():
                for p in net.parameters():
                    if p.requires_grad and p.grad is not None:
                        # 1. Raw analytical update
                        raw_update = -current_lr * p.grad
                        
                        # 2. Stochastic Rounding to Q1.7 grid (1/128)
                        scaled_update = raw_update / DELTA
                        floor_update = torch.floor(scaled_update)
                        prob = scaled_update - floor_update
                        stoch_round = torch.bernoulli(prob.clamp(0.0, 1.0))
                        
                        quantized_update = (floor_update + stoch_round) * DELTA
                        
                        # 3. Read-Modify-Write (Updating the weight)
                        p.data.add_(quantized_update)
                        
                        # 4. Enforce hardware Q1.7 range limits
                        p.data.clamp_(-1.0, 127.0 / 128.0)
            # ─────────────────────────────────────────────
            
            sum_loss += total_loss.item()
            preds = final3.sum(0).argmax(1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        train_acc = 100.0 * correct / total
        scheduler.step()
        test_acc, per_cls = evaluate(net, test_loader)
        gap = train_acc - test_acc
        n = len(train_loader)
        
        print(f"{epoch+1:>4} | {sum_loss/n:>9.4f} | {train_acc:>6.2f}% | {test_acc:>6.2f}% | {gap:>+5.1f}% | {current_lr:.2e}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), f"{MODEL_DIR}/snn_dcll_best.pth")

        if (epoch + 1) % 10 == 0:
            print(f"\n     Per-class accuracy (epoch {epoch+1}):")
            for c, name in enumerate(CLASS_NAMES):
                bar = "█" * int(per_cls[c] / 5)
                print(f"       {name:<30} {per_cls[c]:>6.2f}%  {bar}")
            print()
            
    print("=" * 70)
    print(f"Best Test Accuracy (DCLL HW Constraints) : {best_acc:.2f}%")

if __name__ == "__main__":
    train_model()

