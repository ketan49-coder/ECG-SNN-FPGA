import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset

# Import the network architecture from the training script
from train_snn_superspike import ECG_SuperSpike, OUTPUT_CLASSES, CLASS_NAMES

MODEL_PATH = "/content/drive/MyDrive/SNN_Models/snn_superspike_best.pth"
DATA_DIR = "data"
BATCH_SIZE = 128

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading test data...")
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                            torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load Model
    print(f"Loading trained model from {MODEL_PATH}...")
    net = ECG_SuperSpike().to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Make sure training has finished.")
        return

    # Handle older PyTorch versions where weights_only might not exist
    try:
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    except TypeError:
        net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        
    net.eval()

    # 3. Run Inference
    all_preds = []
    all_targets = []

    print("Running inference on test set...")
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            spk_out = net(data)
            preds = spk_out.sum(0).argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # 4. Generate Metrics
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (Precision, Recall, F1-Score)")
    print("="*60)
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4))

    # 5. Save Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix: 32-Neuron SuperSpike SNN')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path)
    print(f"\nConfusion matrix plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    evaluate_model()
