#  MPC-Based Online Learning Control for Induction Heated Reactor

This repository contains the implementation of a **Model Predictive Control (MPC)** framework integrated with **online learning neural network models** (LSTM and ESN) for dynamic prediction and control of an induction-heated reactor system.

The project focuses on real-time prediction, adaptive learning, and optimal control under constrained operating conditions.

---

## 📂 Project Structure

### 📓 Main Notebook
- **MPC_online_ESN_LSTM_02.ipynb**
  - Main experimental notebook
  - Demonstrates workflow including:
    - Data preparation
    - Model training
    - Online learning
    - MPC simulation
    - Performance visualization

---

### 🧠 Online Learning Models

#### 1. `LSTM_online.py`
- Online learning implementation using LSTM (TensorFlow/Keras)
- Features:
  - Initial offline training
  - Online batch updates
  - State extraction
  - Prediction module
- Used for dynamic time-series prediction of reactor behavior

#### 2. `ESN_online_01.py`
- Echo State Network (ESN) implementation
- Reservoir computing-based online learning model
- Designed for:
  - Fast incremental learning
  - Real-time prediction updates
  - Dynamic system modeling

---

### ⚙️ Reactor Modeling

#### 3. `ModelReactor_class.py`
- Reactor dynamic simulation class
- Simulates reactor response to control inputs
- Used inside MPC prediction horizon

#### 4. `Model_def.py`
- Supporting model definitions
- Functions and utilities for reactor modeling
- May include system equations or helper functions

#### 5. `Model_constant.py`
- Contains model constants such as:
  - Physical parameters
  - Operating limits
  - Model coefficients
- Centralized parameter management

---

### 🎯 Model Predictive Control

#### 6. `MPC_ModelReactor.py`
- MPC implementation for reactor temperature control
- Features:
  - Prediction horizon simulation
  - Control horizon optimization
  - Cost function with tracking and control penalties
  - Constraints on control movement
  - Optimization methods:
    - SLSQP
    - Differential Evolution
- Integrates reactor simulation with optimization routine

---

## ⚙️ Dependencies

- Python 3.x
- TensorFlow / Keras
- NumPy
- Pandas
- SciPy
- Matplotlib
- Joblib
- OpenPyXL

Install dependencies:

```bash
pip install numpy pandas scipy matplotlib tensorflow joblib openpyxl
