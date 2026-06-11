import numpy as np
from generate_data import build_datasets
from models import RISFaultMLP
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

print("Testing more samples...")
X_real, X_complex, y_faulty, y_element = build_datasets(n_samples_per_class=1000, snr_db=30)
max_val = np.max(np.abs(X_complex))
scale_factor = max_val * 1.15
X_complex_scaled = X_complex / scale_factor

def preprocess(complex_data):
    flat = np.empty((complex_data.shape[0], 4 * 2), dtype=float)
    flat[:, 0::2] = np.tanh(np.real(complex_data))
    flat[:, 1::2] = np.tanh(np.imag(complex_data))
    return flat

X = preprocess(X_complex_scaled).astype(np.float32)
y = y_faulty.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(8, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(50):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} loss: {loss.item()}")
        
model.eval()
with torch.no_grad():
    outputs = model(torch.tensor(X_test))
    _, predicted = torch.max(outputs, 1)
    acc = accuracy_score(y_test, predicted.numpy())
    print(f"Det Acc: {acc}")
