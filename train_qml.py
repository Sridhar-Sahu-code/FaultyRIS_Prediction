import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import time

# ==========================================================
# ── HYBRID QML MODEL (PennyLane + PyTorch) ────────────────
# ==========================================================
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class RISFaultQML(nn.Module):
    def __init__(self, num_classes=16, n_layers=1):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
        # Classical bypass to ensure state-of-the-art accuracy
        self.classical_bypass = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc = nn.Linear(n_qubits + 32, num_classes)
        
    def forward(self, x):
        q_out = self.q_layer(x)
        c_out = self.classical_bypass(x)
        combined = torch.cat((q_out, c_out), dim=1)
        return self.fc(combined)

def train_and_evaluate(model_class, num_classes, train_loader, test_loader, epochs, lr, model_name, task_name):
    # QML usually runs on CPU with default.qubit
    device = torch.device("cpu")
    print(f"\n--- Training {model_name} for {task_name} on {device} ---")
    
    model = model_class(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
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
        
        if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")
            
    training_time = time.time() - start_time
    print(f"Training finished in {training_time:.2f} seconds.")
    
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

def run_pipeline():
    os.makedirs("results", exist_ok=True)
    
    if not os.path.exists("data/X_real.npy"):
        print("Dataset not found. Generating dataset first...")
        os.system("python generate_data.py")

    X = np.load("data/X_real.npy").astype(np.float32)
    y_faulty = np.load("data/y_faulty.npy").astype(np.int64)
    y_element = np.load("data/y_element.npy").astype(np.int64)
    
    batch_size = 64
    epochs = 15  # QML is slow, keep epochs low
    lr = 0.01

    print("\n" + "="*60)
    print(" TASK 1: BINARY FAULT DETECTION (Perfect vs Faulty)")
    print("="*60)
    
    X_train_det, X_test_det, y_train_det, y_test_det = train_test_split(
        X, y_faulty, test_size=0.20, random_state=42, stratify=y_faulty
    )
    
    train_dataset_det = TensorDataset(torch.tensor(X_train_det), torch.tensor(y_train_det))
    test_dataset_det = TensorDataset(torch.tensor(X_test_det), torch.tensor(y_test_det))
    train_loader_det = DataLoader(train_dataset_det, batch_size=batch_size, shuffle=True)
    test_loader_det = DataLoader(test_dataset_det, batch_size=batch_size, shuffle=False)
    
    qml_det_metrics = train_and_evaluate(
        RISFaultQML, num_classes=2, train_loader=train_loader_det, test_loader=test_loader_det,
        epochs=epochs, lr=lr, model_name="Hybrid QML", task_name="Fault Detection"
    )
    
    print("\n" + "="*60)
    print(" TASK 2: 16-CLASS FAULT LOCALIZATION (Locate Faulty Element)")
    print("="*60)
    
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
    
    qml_loc_metrics = train_and_evaluate(
        RISFaultQML, num_classes=16, train_loader=train_loader_loc, test_loader=test_loader_loc,
        epochs=epochs, lr=lr, model_name="Hybrid QML", task_name="Fault Localization"
    )
    
    print("\n" + "="*60)
    print("                     QML PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Model':<15} | {'Det Acc':<8} | {'Det F1':<8} | {'Loc Acc':<8} | {'Loc F1':<8}")
    print("-"*60)
    print(f"{'Hybrid QML':<15} | {qml_det_metrics['accuracy']*100:>7.2f}% | {qml_det_metrics['f1']*100:>7.2f}% | {qml_loc_metrics['accuracy']*100:>7.2f}% | {qml_loc_metrics['f1']*100:>7.2f}%")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
