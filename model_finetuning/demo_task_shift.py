"""
Transfer Learning with PyTorch: Demonstrating Task Change with the Same Features but Different Targets

This script illustrates transfer learning where the input features remain the same,
but the target task changes. We simulate two tasks:
1. Predicting hypertension (task 1) using adult data
2. Predicting obesity risk (task 2) using youth data

Key steps:
- Pretrain a model on task 1 (hypertension prediction)
- Freeze the feature extractor (body)
- Fine-tune the classification head on task 2 (obesity risk prediction)
- Evaluate performance on both tasks to demonstrate transfer learning across tasks

This differs from domain adaptation where the task remains the same but the data distribution changes.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
import random

# =====================================================================
# SET SEED FOR REPRODUCIBILITY
# =====================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =====================================================================
# DATA GENERATION
# =====================================================================

n_adults = 1000
n_youth = 300

def simulate_data_task_change(n, age_range):
    """
    Generate synthetic medical data with same features but two different targets:
    - For adults: hypertension diagnosis (task 1)
    - For youth: obesity risk (task 2)

    Args:
        n (int): number of samples
        age_range (tuple): (min_age, max_age)

    Returns:
        DataFrame with features and both targets (NaN for unused target)
    """
    age = np.random.randint(age_range[0], age_range[1], n)
    bmi = np.random.normal(25 if age_range[0] > 20 else 20, 3, n)
    systolic_bp = np.random.normal(130 if age_range[0] > 20 else 120, 10 if age_range[0] > 20 else 15, n)
    cholesterol = np.random.normal(200 if age_range[0] > 20 else 210, 25, n)
    smoker = np.random.randint(0, 2, n)

    data = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'cholesterol': cholesterol,
        'smoker': smoker
    })

    # Task 1 target: hypertension (adults)
    if age_range[0] > 20:
        bp_threshold = 140
        chol_threshold = 240
        data['hypertension'] = ((systolic_bp > bp_threshold) | (cholesterol > chol_threshold)).astype(int)
        data['obesity_risk'] = np.nan  # No obesity risk label here
    else:
        # Task 2 target: obesity risk (youth)
        # Define obesity risk as BMI > 25 or smoker
        data['obesity_risk'] = ((bmi > 25) | (smoker == 1)).astype(int)
        data['hypertension'] = np.nan  # No hypertension label here

    return data

adult_data = simulate_data_task_change(n_adults, (30, 65))
youth_data = simulate_data_task_change(n_youth, (13, 20))

# =====================================================================
# PREPARE DATA FOR TASK 1 (HYPERTENSION PREDICTION)
# =====================================================================

X_adult = adult_data.drop(['hypertension', 'obesity_risk'], axis=1)
y_adult = adult_data['hypertension']

X_train_adult, X_val_adult, y_train_adult, y_val_adult = train_test_split(
    X_adult, y_adult, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_adult_scaled = scaler.fit_transform(X_train_adult)
X_val_adult_scaled = scaler.transform(X_val_adult)

# =====================================================================
# PREPARE DATA FOR TASK 2 (OBESITY RISK PREDICTION)
# =====================================================================

X_youth = youth_data.drop(['hypertension', 'obesity_risk'], axis=1)
y_youth = youth_data['obesity_risk']

X_train_youth, X_val_youth, y_train_youth, y_val_youth = train_test_split(
    X_youth, y_youth, test_size=0.2, random_state=42
)

# Use the SAME scaler fitted on adult data to maintain consistent feature scaling
X_train_youth_scaled = scaler.transform(X_train_youth)
X_val_youth_scaled = scaler.transform(X_val_youth)

# =====================================================================
# MODEL DEFINITION
# =====================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with separate body and head for transfer learning across tasks.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        x = self.body(x)
        return torch.sigmoid(self.head(x))

def to_tensor(X, y):
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y.values, dtype=torch.float32).view(-1,1))

X_train_adult_tensor, y_train_adult_tensor = to_tensor(X_train_adult_scaled, y_train_adult)
X_val_adult_tensor, y_val_adult_tensor = to_tensor(X_val_adult_scaled, y_val_adult)

X_train_youth_tensor, y_train_youth_tensor = to_tensor(X_train_youth_scaled, y_train_youth)
X_val_youth_tensor, y_val_youth_tensor = to_tensor(X_val_youth_scaled, y_val_youth)

# =====================================================================
# TRAIN ON TASK 1 (HYPERTENSION PREDICTION)
# =====================================================================

model = MLP(input_dim=X_train_adult_scaled.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_model(model, X, y, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

print("Pretraining on Task 1: Hypertension prediction (adult data)...")
train_model(model, X_train_adult_tensor, y_train_adult_tensor)

# =====================================================================
# EVALUATE ON TASK 1
# =====================================================================

model.eval()
with torch.no_grad():
    val_preds_task1 = model(X_val_adult_tensor).round()
    precision1, recall1, f1_1, _ = precision_recall_fscore_support(
        y_val_adult_tensor, val_preds_task1, average='binary'
    )

print("\nTask 1 (Hypertension) Validation Performance:")
print(f"Precision: {precision1:.4f}")
print(f"Recall:    {recall1:.4f}")
print(f"F1 Score:  {f1_1:.4f}")

# =====================================================================
# TRANSFER LEARNING: FREEZE BODY, FINE-TUNE HEAD ON TASK 2
# =====================================================================

print("\nFreezing model body and fine-tuning head on Task 2: Obesity risk prediction (youth data)...")
for param in model.body.parameters():
    param.requires_grad = False

# Replace head with new randomly initialized layer for new task
model.head = nn.Linear(16, 1)

# New optimizer only for head parameters
optimizer = optim.Adam(model.head.parameters(), lr=0.01)

train_model(model, X_train_youth_tensor, y_train_youth_tensor, epochs=20)

# =====================================================================
# EVALUATE ON TASK 2
# =====================================================================

model.eval()
with torch.no_grad():
    val_preds_task2 = model(X_val_youth_tensor).round()
    precision2, recall2, f1_2, _ = precision_recall_fscore_support(
        y_val_youth_tensor, val_preds_task2, average='binary'
    )

print("\nTask 2 (Obesity Risk) Validation Performance:")
print(f"Precision: {precision2:.4f}")
print(f"Recall:    {recall2:.4f}")
print(f"F1 Score:  {f1_2:.4f}")

# =====================================================================
# EVALUATE MODEL ON TASK 1 AGAIN TO SEE IF BODY FREEZING PRESERVED KNOWLEDGE
# =====================================================================

# To evaluate on task 1 again, we need to temporarily replace head with original head trained on task 1
# But original head was replaced. So instead, we create a fresh head and train it on task 1 data only on frozen body
# This simulates how well the frozen body still supports task 1

print("\nEvaluating frozen body on Task 1 with new head trained on Task 1 data...")

# Freeze body remains, replace head
model.head = nn.Linear(16, 1)
optimizer = optim.Adam(model.head.parameters(), lr=0.01)

# Train head on task 1 data only (body frozen)
train_model(model, X_train_adult_tensor, y_train_adult_tensor, epochs=20)

model.eval()
with torch.no_grad():
    val_preds_task1_after = model(X_val_adult_tensor).round()
    precision1_after, recall1_after, f1_1_after, _ = precision_recall_fscore_support(
        y_val_adult_tensor, val_preds_task1_after, average='binary'
    )

print("\nTask 1 (Hypertension) Validation Performance after fine-tuning head on Task 2 and retraining head on Task 1:")
print(f"Precision: {precision1_after:.4f}")
print(f"Recall:    {recall1_after:.4f}")
print(f"F1 Score:  {f1_1_after:.4f}")
