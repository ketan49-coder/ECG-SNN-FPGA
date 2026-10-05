import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

# ==========================================
# ADVANCED "SOTA KILLER" SNN TRAINING PIPELINE
# Upgrades included:
# 1. Direct Current Injection (No LFSR hardware needed)
# 2. Focal Loss (For extreme minority class learning)
# 3. Quantization-Aware Training (QAT) with Straight-Through Estimator
# ==========================================

# --- Hyperparameters ---
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
NUM_STEPS = 16
BETA = 0.9375     # Strictly maps to >> 4 bit-shift in hardware
THRESHOLD = 0.5   # Lower threshold for deep SNN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Quantization-Aware Training (Straight-Through Estimator) ---
class Q1_7_Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weights):
        clamped = torch.clamp(weights, -1.0, 0.9921875)
        quantized = torch.round(clamped * 128.0) / 128.0
        return quantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def quantize_q17(weights):
    return Q1_7_Quantize.apply(weights)

# --- 2. Advanced Architecture (Direct Injection + QAT) ---
class ECG_SNN_SOTA(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc1 = nn.Linear(128, 16, bias=False)
        self.lif1 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")
        
        self.fc2 = nn.Linear(16, 16, bias=False)
        self.lif2 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")
        
        self.fc3 = nn.Linear(16, 5, bias=False)
        self.lif3 = snn.Leaky(beta=BETA, threshold=THRESHOLD, spike_grad=spike_grad, reset_mechanism="zero")

    def forward(self, x):
        mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        spk3_rec = []
        
        for step in range(NUM_STEPS):
            # UPGRADE 1 & 3: Direct Current Injection + QAT weights
            cur1 = F.linear(x, quantize_q17(self.fc1.weight))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = F.linear(spk1, quantize_q17(self.fc2.weight))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = F.linear(spk2, quantize_q17(self.fc3.weight))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk3_rec.append(spk3)
            
        return torch.stack(spk3_rec, dim=0)

# --- 3. Focal Loss for Extreme Class Imbalance ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

