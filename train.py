import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os
import time

# Import models
from models import RISFaultMLP, RISFaultCNN, RISFaultQML

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================================
# ── LOADING DATA ──────────────────────────────────────────
# ==========================================================
if not os.path.exists("data/X_real.npy"):
    print("Dataset not found. Generating dataset first...")
    os.system("python generate_data.py")

X = np.load("data/X_real.npy").astype(np.float32)
y_faulty = np.load("data/y_faulty.npy").astype(np.int64)
y_element = np.load("data/y_element.npy").astype(np.int64)

# ==========================================================
# ── TRAINING HELPER FUNCTION ──────────────────────────────
# ==========================================================
def train_and_evaluate(model_class, num_classes, train_loader, test_loader, epochs, lr, model_name, task_name):
    print(f"\n--- Training {model_name} for {task_name} ---")
    
    # Instantiate model
    model = model_class(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    history = {'loss': [], 'acc': []}
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_x.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")
            
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds.")
    
    # Evaluation on Test set
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'precision': precision_score(all_targets, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_targets, all_preds, average='macro', zero_division=0),
        'f1': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'history': history,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return metrics

# ==========================================================
# ── MAIN PIPELINE ─────────────────────────────────────────
# ==========================================================
def run_pipeline():
    os.makedirs("results", exist_ok=True)
    
    # ------------------------------------------------------
    # Task 1: Binary Fault Detection (Perfect vs Faulty)
    # ------------------------------------------------------
    print("\n" + "="*60)
    print(" TASK 1: BINARY FAULT DETECTION (Perfect vs Faulty)")
    print("="*60)
    
    X_train_det, X_test_det, y_train_det, y_test_det = train_test_split(
        X, y_faulty, test_size=0.20, random_state=42, stratify=y_faulty
    )
    
    # Create DataLoader
    batch_size = 64
    train_dataset_det = TensorDataset(torch.tensor(X_train_det), torch.tensor(y_train_det))
    test_dataset_det = TensorDataset(torch.tensor(X_test_det), torch.tensor(y_test_det))
    
    train_loader_det = DataLoader(train_dataset_det, batch_size=batch_size, shuffle=True)
    test_loader_det = DataLoader(test_dataset_det, batch_size=batch_size, shuffle=False)
    
    # Hyperparameters for Binary Detection
    epochs_classical = 20
    epochs_qml = 8  # Keep it slightly lower to ensure fast runtime on CPU
    lr_classical = 0.005
    lr_qml = 0.01
    
    # Train MLP
    mlp_det_metrics = train_and_evaluate(
        RISFaultMLP, num_classes=2, train_loader=train_loader_det, test_loader=test_loader_det,
        epochs=epochs_classical, lr=lr_classical, model_name="Classical MLP", task_name="Fault Detection"
    )
    
    # Train CNN
    cnn_det_metrics = train_and_evaluate(
        RISFaultCNN, num_classes=2, train_loader=train_loader_det, test_loader=test_loader_det,
        epochs=epochs_classical, lr=lr_classical, model_name="Classical CNN", task_name="Fault Detection"
    )
    
    # Train QML
    qml_det_metrics = train_and_evaluate(
        RISFaultQML, num_classes=2, train_loader=train_loader_det, test_loader=test_loader_det,
        epochs=epochs_qml, lr=lr_qml, model_name="Hybrid QML", task_name="Fault Detection"
    )
    
    # ------------------------------------------------------
    # Task 2: 16-Class Fault Localization (Locate Faulty Element)
    # ------------------------------------------------------
    print("\n" + "="*60)
    print(" TASK 2: 16-CLASS FAULT LOCALIZATION (Locate Faulty Element)")
    print("="*60)
    
    # Filter for faulty samples only
    faulty_mask = (y_faulty == 1)
    X_faulty = X[faulty_mask]
    y_element_faulty = y_element[faulty_mask]
    
    X_train_loc, X_test_loc, y_train_loc, y_test_loc = train_test_split(
        X_faulty, y_element_faulty, test_size=0.20, random_state=42, stratify=y_element_faulty
    )
    
    train_dataset_loc = TensorDataset(torch.tensor(X_train_loc), torch.tensor(y_train_loc))
    test_dataset_loc = TensorDataset(torch.tensor(X_test_loc), torch.tensor(y_test_loc))
    
    train_loader_loc = DataLoader(train_dataset_loc, batch_size=batch_size, shuffle=True)
    test_loader_loc = DataLoader(test_dataset_loc, batch_size=batch_size, shuffle=False)
    
    # Train MLP
    mlp_loc_metrics = train_and_evaluate(
        RISFaultMLP, num_classes=16, train_loader=train_loader_loc, test_loader=test_loader_loc,
        epochs=epochs_classical, lr=lr_classical, model_name="Classical MLP", task_name="Fault Localization"
    )
    
    # Train CNN
    cnn_loc_metrics = train_and_evaluate(
        RISFaultCNN, num_classes=16, train_loader=train_loader_loc, test_loader=test_loader_loc,
        epochs=epochs_classical, lr=lr_classical, model_name="Classical CNN", task_name="Fault Localization"
    )
    
    # Train QML
    qml_loc_metrics = train_and_evaluate(
        RISFaultQML, num_classes=16, train_loader=train_loader_loc, test_loader=test_loader_loc,
        epochs=epochs_qml, lr=lr_qml, model_name="Hybrid QML", task_name="Fault Localization"
    )
    
    # ------------------------------------------------------
    # REPORT AND EXPORTS
    # ------------------------------------------------------
    print("\n" + "="*60)
    print("                     FINAL PERFORMANCE COMPARISON")
    print("="*60)
    print(f"{'Model':<15} | {'Det Acc':<8} | {'Det F1':<8} | {'Loc Acc':<8} | {'Loc F1':<8}")
    print("-"*60)
    print(f"{'Classical MLP':<15} | {mlp_det_metrics['accuracy']*100:>7.2f}% | {mlp_det_metrics['f1']*100:>7.2f}% | {mlp_loc_metrics['accuracy']*100:>7.2f}% | {mlp_loc_metrics['f1']*100:>7.2f}%")
    print(f"{'Classical CNN':<15} | {cnn_det_metrics['accuracy']*100:>7.2f}% | {cnn_det_metrics['f1']*100:>7.2f}% | {cnn_loc_metrics['accuracy']*100:>7.2f}% | {cnn_loc_metrics['f1']*100:>7.2f}%")
    print(f"{'Hybrid QML':<15} | {qml_det_metrics['accuracy']*100:>7.2f}% | {qml_det_metrics['f1']*100:>7.2f}% | {qml_loc_metrics['accuracy']*100:>7.2f}% | {qml_loc_metrics['f1']*100:>7.2f}%")
    print("="*60)
    
    # Save a comparison summary text file
    with open("results/performance_comparison.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("                     FINAL PERFORMANCE COMPARISON\n")
        f.write("="*60 + "\n")
        f.write(f"{'Model':<15} | {'Det Acc':<8} | {'Det F1':<8} | {'Loc Acc':<8} | {'Loc F1':<8}\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Classical MLP':<15} | {mlp_det_metrics['accuracy']*100:>7.2f}% | {mlp_det_metrics['f1']*100:>7.2f}% | {mlp_loc_metrics['accuracy']*100:>7.2f}% | {mlp_loc_metrics['f1']*100:>7.2f}%\n")
        f.write(f"{'Classical CNN':<15} | {cnn_det_metrics['accuracy']*100:>7.2f}% | {cnn_det_metrics['f1']*100:>7.2f}% | {cnn_loc_metrics['accuracy']*100:>7.2f}% | {cnn_loc_metrics['f1']*100:>7.2f}%\n")
        f.write(f"{'Hybrid QML':<15} | {qml_det_metrics['accuracy']*100:>7.2f}% | {qml_det_metrics['f1']*100:>7.2f}% | {qml_loc_metrics['accuracy']*100:>7.2f}% | {qml_loc_metrics['f1']*100:>7.2f}%\n")
        f.write("="*60 + "\n")
        
    # Plot learning curves
    plt.figure(figsize=(14, 10))
    
    # Loss curves for detection
    plt.subplot(2, 2, 1)
    plt.plot(mlp_det_metrics['history']['loss'], label='MLP')
    plt.plot(cnn_det_metrics['history']['loss'], label='CNN')
    plt.plot(qml_det_metrics['history']['loss'], label='QML')
    plt.title('Fault Detection Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy curves for detection
    plt.subplot(2, 2, 2)
    plt.plot(mlp_det_metrics['history']['acc'], label='MLP')
    plt.plot(cnn_det_metrics['history']['acc'], label='CNN')
    plt.plot(qml_det_metrics['history']['acc'], label='QML')
    plt.title('Fault Detection Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Loss curves for localization
    plt.subplot(2, 2, 3)
    plt.plot(mlp_loc_metrics['history']['loss'], label='MLP')
    plt.plot(cnn_loc_metrics['history']['loss'], label='CNN')
    plt.plot(qml_loc_metrics['history']['loss'], label='QML')
    plt.title('Fault Localization Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy curves for localization
    plt.subplot(2, 2, 4)
    plt.plot(mlp_loc_metrics['history']['acc'], label='MLP')
    plt.plot(cnn_loc_metrics['history']['acc'], label='CNN')
    plt.plot(qml_loc_metrics['history']['acc'], label='QML')
    plt.title('Fault Localization Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("results/learning_curves.png")
    print("\nSaved learning curves plot to results/learning_curves.png")
    plt.close()

if __name__ == "__main__":
    run_pipeline()
