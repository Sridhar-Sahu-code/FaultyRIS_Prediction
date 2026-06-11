import torch
import torch.nn as nn
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
import pennylane as qml
import numpy as np

# ==========================================================
# ── CLASSICAL MLP MODEL ───────────────────────────────────
# ==========================================================
class RISFaultMLP(nn.Module):
    def __init__(self, input_dim=8, num_classes=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================================
# ── CLASSICAL CNN MODEL ───────────────────────────────────
# ==========================================================
class RISFaultCNN(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Reshaped input is (batch, 2, 4). After two Conv1d with padding=1, size is still (batch, 32, 4)
        # Flattened size: 32 * 4 = 128
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # x is (batch, 8) -> reshape to (batch, channels=2, length=4)
        # real parts are elements 0::2, imag parts are elements 1::2
        x = x.view(-1, 2, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # Flatten to (batch, 128)
        return self.fc(x)

# ==========================================================
# ── HYBRID QML MODEL (PennyLane + PyTorch) ────────────────
# ==========================================================
n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # inputs are in [-1, 1], map to [-pi, pi]
    qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class RISFaultQML(nn.Module):
    def __init__(self, num_classes=16, n_layers=2):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.classical_fc = nn.Linear(n_qubits, num_classes)
        
    def forward(self, x):
        q_out = self.q_layer(x)
        return self.classical_fc(q_out)
