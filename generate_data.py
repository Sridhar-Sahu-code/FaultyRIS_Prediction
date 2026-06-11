import numpy as np
import pandas as pd
import os

# ── System Parameters ─────────────────────────────────────────
N_Tx = 4
N_RIS = 8
frequency = 26e9
velocity_of_light = 3e8
lambda_c = velocity_of_light / frequency
kappa = 2.2
path_loss_ref_linear = 10**(-20/10)
rician_factor_linear = 10**(5/10)
d_spacing = 0.5 * lambda_c

loc_BS_center   = np.array([0,   0, 15])
loc_RIS1_center = np.array([10, -5,  5])
loc_RIS2_center = np.array([10,  5,  5])
loc_User        = np.array([15,  0,  1])

varpi_ref = 0.5
zeta_ref = np.pi / 4
Upsilon_ref = np.diag(np.sqrt(varpi_ref) * np.exp(1j * zeta_ref) * np.ones(N_RIS))

def get_array_elements(center, n, spacing, orient='horizontal'):
    elems = np.zeros((n, 3))
    start = -((n - 1) * spacing) / 2
    for i in range(n):
        pos = np.copy(center)
        if orient == 'horizontal': 
            pos[1] += start + (i * spacing)
        else:                    
            pos[2] += start + (i * spacing)
        elems[i] = pos
    return elems

pos_BS       = get_array_elements(loc_BS_center,   N_Tx,  d_spacing, 'horizontal')
pos_RIS1     = get_array_elements(loc_RIS1_center, N_RIS, d_spacing, 'vertical')
pos_RIS2     = get_array_elements(loc_RIS2_center, N_RIS, d_spacing, 'vertical')
pos_User_arr = np.array([loc_User])

def create_spatial_correlation_matrix(positions, lambda_c):
    n = len(positions)
    Omega = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            d = np.linalg.norm(positions[i] - positions[j])
            arg = 2 * np.pi * d / lambda_c
            Omega[i, j] = 1.0 if arg == 0 else np.sin(arg) / arg
            if i == j: 
                Omega[i, j] += 1e-9
    ev = np.min(np.real(np.linalg.eigvals(Omega)))
    if ev < 0: 
        Omega -= (ev - 1e-9) * np.eye(n)
    return np.linalg.cholesky(Omega)

L_Omega_RIS1 = create_spatial_correlation_matrix(pos_RIS1, lambda_c)
L_Omega_RIS2 = create_spatial_correlation_matrix(pos_RIS2, lambda_c)

def generate_near_field_channel(tx_pos, rx_pos, L_Omega_rx=None, L_Omega_tx=None):
    nTx = len(tx_pos)
    nRx = len(rx_pos)
    H_LoS = np.zeros((nRx, nTx), dtype=complex)
    for r in range(nRx):
        for t in range(nTx):
            d = np.linalg.norm(rx_pos[r] - tx_pos[t])
            H_LoS[r, t] = np.sqrt(path_loss_ref_linear * (d**-kappa)) * np.exp(-1j * 2 * np.pi * d / lambda_c)
    noise_H = (np.random.randn(nRx, nTx) + 1j * np.random.randn(nRx, nTx)) / np.sqrt(2)
    if L_Omega_rx is not None: 
        noise_H = L_Omega_rx @ noise_H
    if L_Omega_tx is not None: 
        noise_H = noise_H @ L_Omega_tx.T
    avg_d = np.linalg.norm(np.mean(rx_pos, axis=0) - np.mean(tx_pos, axis=0))
    avg_pl = np.sqrt(path_loss_ref_linear * (avg_d**-kappa))
    Z = rician_factor_linear
    return (np.sqrt(Z / (Z + 1)) * H_LoS) + (np.sqrt(1 / (Z + 1)) * avg_pl * noise_H)

H1_fixed = generate_near_field_channel(pos_BS, pos_RIS1, L_Omega_rx=L_Omega_RIS1)
g1_fixed = generate_near_field_channel(pos_RIS1, pos_User_arr, L_Omega_tx=L_Omega_RIS1)
H2_fixed = generate_near_field_channel(pos_BS, pos_RIS2, L_Omega_rx=L_Omega_RIS2)
g2_fixed = generate_near_field_channel(pos_RIS2, pos_User_arr, L_Omega_tx=L_Omega_RIS2)

def generate_multi_ris_realization_with_fault(N_pilot, fault_idx=None, snr_db=30):
    """
    fault_idx: None for Perfect, 0-7 for faulty RIS1 element, 8-15 for faulty RIS2 element
    """
    H1 = H1_fixed
    g1 = g1_fixed
    H2 = H2_fixed
    g2 = g2_fixed
    
    phi1 = np.sqrt(varpi_ref) * np.exp(1j * zeta_ref) * np.ones(N_RIS)
    phi2 = np.sqrt(varpi_ref) * np.exp(1j * zeta_ref) * np.ones(N_RIS)
    
    if fault_idx is not None:
        if fault_idx < N_RIS:
            phi1[fault_idx] = 0.0
        else:
            phi2[fault_idx - N_RIS] = 0.0
            
    U1 = np.diag(phi1)
    U2 = np.diag(phi2)
    
    A_true = (H1.T.conj() @ U1 @ g1.T).flatten() + (H2.T.conj() @ U2 @ g2.T).flatten()
    sp = np.linalg.norm(A_true)**2
    sl = 10**(snr_db/10)
    nstd = np.sqrt(sp / (sl * (N_pilot / 50)) / 2)
    noise = nstd * (np.random.randn(N_Tx) + 1j * np.random.randn(N_Tx))
    return A_true + noise, A_true

def preprocess(complex_data):
    flat = np.empty((complex_data.shape[0], N_Tx * 2), dtype=float)
    flat[:, 0::2] = np.tanh(np.real(complex_data))
    flat[:, 1::2] = np.tanh(np.imag(complex_data))
    return flat

def build_datasets(n_samples_per_class=120, N_pilot=100, snr_db=10):
    """
    Generates balanced datasets:
    1. Detection dataset: 50% Perfect, 50% Faulty (balanced across 16 fault positions).
    2. Localization dataset: 100% Faulty (balanced across 16 fault positions).
    """
    print(f"Generating dataset with {n_samples_per_class} samples per class (17 classes: Perfect + 16 Faulty)...")
    
    complex_inputs = []
    labels_faulty = []   # 0 for perfect, 1 for faulty
    labels_element = []  # -1 for perfect, 0-15 for faulty elements
    
    # Perfect realizations
    for _ in range(n_samples_per_class * 16):
        c, _ = generate_multi_ris_realization_with_fault(N_pilot, fault_idx=None, snr_db=snr_db)
        complex_inputs.append(c)
        labels_faulty.append(0)
        labels_element.append(-1)
        
    # Faulty realizations
    for fault_idx in range(16):
        for _ in range(n_samples_per_class):
            c, _ = generate_multi_ris_realization_with_fault(N_pilot, fault_idx=fault_idx, snr_db=snr_db)
            complex_inputs.append(c)
            labels_faulty.append(1)
            labels_element.append(fault_idx)
            
    complex_inputs = np.array(complex_inputs)
    labels_faulty = np.array(labels_faulty)
    labels_element = np.array(labels_element)
    
    # Preprocess
    real_features = preprocess(complex_inputs)
    
    return real_features, complex_inputs, labels_faulty, labels_element

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # Generate dataset
    X_real, X_complex, y_faulty, y_element = build_datasets(n_samples_per_class=500, snr_db=25)
    
    # Scale factor calculations
    max_val = np.max(np.abs(X_complex))
    scale_factor = max_val * 1.15
    X_complex_scaled = X_complex / scale_factor
    X_real_scaled = preprocess(X_complex_scaled)
    
    # Save datasets
    np.save("data/X_real.npy", X_real_scaled)
    np.save("data/X_complex.npy", X_complex_scaled)
    np.save("data/y_faulty.npy", y_faulty)
    np.save("data/y_element.npy", y_element)
    
    print(f"Data generation complete. Datasets saved in data/ directory.")
    print(f"Features shape: {X_real_scaled.shape}")
    print(f"Faulty labels distribution: {np.bincount(y_faulty)}")
    print(f"Element labels distribution: {np.bincount(y_element[y_element >= 0])}")
