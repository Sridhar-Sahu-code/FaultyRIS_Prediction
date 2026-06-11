# Faulty RIS Element Prediction using Classical & Quantum Machine Learning

![RIS Prediction](https://img.shields.io/badge/Project-6G%20Telecom%20AI-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)

## 📌 Project Overview
Reconfigurable Intelligent Surfaces (RIS) are massive arrays of reflecting elements—acting as "smart mirrors"—that optimize wireless signal propagation in modern 6G networks. However, due to harsh outdoor conditions, individual RIS elements can degrade or fail over time, severely impacting network performance. 

This project simulates a wireless communication environment and leverages **Artificial Intelligence (AI)** to achieve two primary objectives without the need for physical hardware inspections:
1. **Fault Detection:** Determine if a hardware fault exists within the RIS array.
2. **Fault Localization:** Pinpoint the exact location of the broken element out of the 16 available tiles.

To achieve this, we compare standard classical architectures with cutting-edge **Quantum Machine Learning (QML)**.

---

## ✨ Key Features
- **Virtual Physics Simulation**: Custom data generator mimicking a realistic 6G near-field communication channel (`generate_data.py`).
- **Large Dataset Generation**: Capable of generating 16,000+ balanced samples of healthy and faulty complex wireless signals.
- **Signal Preprocessing**: Translates complex numbers (Real + Imaginary) into readable vectors for standard AI architectures.
- **Multi-Model AI Comparison**:
  - **MLP (Multi-Layer Perceptron)**: Classical baseline Neural Network.
  - **CNN (Convolutional Neural Network)**: Spatial pattern recognition.
  - **QML (Quantum Machine Learning)**: Hybrid quantum-classical approach.

---

## 🛠️ Prerequisites & Installation
Ensure you have Python 3.8+ installed. You will need the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib torch
```
*(If you are testing QML elements, ensure any required quantum backend simulator libraries like PennyLane/Qiskit are installed as defined in `models.py`)*

---

## 🚀 How to Run

### 1. Generate the Dataset
Before training, you must simulate the wireless environment to create the training data. Run the data generator:
```bash
python generate_data.py
```
> **Note:** This will generate 16,000 samples (8,000 perfect, 8,000 faulty) and save them inside the `data/` directory.

### 2. Train and Evaluate the AI Models
Once the data is generated, run the training pipeline. This script trains the MLP, CNN, and QML models on both tasks (Detection & Localization) and compares their accuracy.
```bash
python train.py
```

### 3. Review Results
Upon completion, the code will output:
- A text file `results/performance_comparison.txt` containing the Accuracy and F1-Scores for all models.
- A visualization `results/learning_curves.png` showing the training loss and accuracy over time.

---

## 📂 Repository Structure

```text
📁 Faulty_RIS_Element_Prediction/
│
├── generate_data.py          # Physics engine & dataset generator
├── models.py                 # PyTorch definitions for MLP, CNN, and QML models
├── train.py                  # Main training loop and evaluation pipeline
├── create_ppt.py             # Script to programmatically generate presentation
├── create_detailed_ppt.py    # Script generating the detailed 23-page presentation
├── data/                     # Generated datasets (.npy files)
├── ppt_assets/               # Diagrams and images used in the presentation
└── README.md                 # Project documentation
```

---

## 🔬 How it Works

### Task 1: Fault Detection (Binary Classification)
The AI receives a complex wireless signal and must determine if the RIS array is operating perfectly or if a fault exists anywhere in the system.

### Task 2: Fault Localization (16-Class Classification)
If a fault is detected, the AI then filters the data and predicts exactly *which* of the 16 individual elements in the array has failed by analyzing the unique distortion pattern in the received signal.

---

## 📝 License & Contact
Created for research and academic demonstration in the intersection of Artificial Intelligence, Quantum Computing, and Telecommunications.