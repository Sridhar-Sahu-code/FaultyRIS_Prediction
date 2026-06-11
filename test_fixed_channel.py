import numpy as np
from generate_data import pos_BS, pos_RIS1, pos_RIS2, pos_User_arr, L_Omega_RIS1, L_Omega_RIS2, generate_near_field_channel, varpi_ref, zeta_ref, N_RIS, N_Tx
from models import RISFaultMLP
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Generate channels ONCE
H1_fixed = generate_near_field_channel(pos_BS, pos_RIS1, L_Omega_rx=L_Omega_RIS1)
g1_fixed = generate_near_field_channel(pos_RIS1, pos_User_arr, L_Omega_tx=L_Omega_RIS1)
H2_fixed = generate_near_field_channel(pos_BS, pos_RIS2, L_Omega_rx=L_Omega_RIS2)
g2_fixed = generate_near_field_channel(pos_RIS2, pos_User_arr, L_Omega_tx=L_Omega_RIS2)

def generate_fixed_channel_sample(fault_idx=None, snr_db=10, N_pilot=100):
    phi1 = np.sqrt(varpi_ref) * np.exp(1j * zeta_ref) * np.ones(N_RIS)
    phi2 = np.sqrt(varpi_ref) * np.exp(1j * zeta_ref) * np.ones(N_RIS)
    
    if fault_idx is not None:
        if fault_idx < N_RIS:
            phi1[fault_idx] = 0.0
        else:
            phi2[fault_idx - N_RIS] = 0.0
            
    U1 = np.diag(phi1)
    U2 = np.diag(phi2)
    
    A_true = (H1_fixed.T.conj() @ U1 @ g1_fixed.T).flatten() + (H2_fixed.T.conj() @ U2 @ g2_fixed.T).flatten()
    sp = np.linalg.norm(A_true)**2
    sl = 10**(snr_db/10)
    nstd = np.sqrt(sp / (sl * (N_pilot / 50)) / 2)
    noise = nstd * (np.random.randn(N_Tx) + 1j * np.random.randn(N_Tx))
    return A_true + noise

complex_inputs = []
labels_faulty = []
for _ in range(1000):
    complex_inputs.append(generate_fixed_channel_sample(None, 20))
    labels_faulty.append(0)
for fault_idx in range(16):
    for _ in range(1000 // 16):
        complex_inputs.append(generate_fixed_channel_sample(fault_idx, 20))
        labels_faulty.append(1)

complex_inputs = np.array(complex_inputs)
labels_faulty = np.array(labels_faulty)

def preprocess(complex_data):
    flat = np.empty((complex_data.shape[0], 4 * 2), dtype=float)
    flat[:, 0::2] = np.tanh(np.real(complex_data))
    flat[:, 1::2] = np.tanh(np.imag(complex_data))
    return flat

X_complex_scaled = complex_inputs / (np.max(np.abs(complex_inputs)) * 1.15)
X = preprocess(X_complex_scaled).astype(np.float32)
y = labels_faulty.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
model.eval()
with torch.no_grad():
    outputs = model(torch.tensor(X_test))
    _, predicted = torch.max(outputs, 1)
    acc = accuracy_score(y_test, predicted.numpy())
    print(f"Fixed Channel Det Acc: {acc}")
