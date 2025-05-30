
"""
Transfer Learning with PyTorch: Fine-tuning a Neural Network for Different Demographics

This script demonstrates a fundamental transfer learning approach using a simple Multi-Layer Perceptron (MLP).
It simulates a medical scenario where we have:
1. A larger dataset for adults (source domain)
2. A smaller dataset for youth (target domain)

The key steps demonstrated are:
- Training a base model on the adult data
- Freezing the feature extraction layers (body)
- Fine-tuning only the classification layer (head) on youth data
- Comparing performance before and after fine-tuning

This approach is particularly useful when:
- You have limited data for your target domain
- The source and target domains are related but different
- You want to leverage knowledge from a data-rich domain
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
# DATA PREPARATION
# =====================================================================

def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries used.

    This ensures that running the code multiple times will produce the same results,
    which is crucial for debugging, testing, and scientific reproducibility.

    Args:
        seed (int): The random seed value to use. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # These settings ensure deterministic behavior at the cost of some performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seeds for reproducibility
set_seed(42)

# Define dataset sizes
n_adults = 1000  # Larger dataset (source domain)
n_youth = 300    # Smaller dataset (target domain)

def simulate_data(n, age_range):
    """
    Generate synthetic medical data with age-appropriate parameters.

    This function creates different data distributions for adults vs. youth,
    simulating real-world differences in medical parameters between age groups.

    Args:
        n (int): Number of samples to generate
        age_range (tuple): (min_age, max_age) for the population

    Returns:
        pandas.DataFrame: Simulated medical data with features and target variable
    """
    # Different physiological parameters based on age group
    if age_range[0] <= 20:  # Youth population
        # Youth-specific parameters with different distributions
        systolic_bp = np.random.normal(120, 15, n)  # Lower baseline BP for youth
        cholesterol = np.random.normal(210, 25, n)  # Different cholesterol distribution
        bp_threshold = 130  # Lower threshold for hypertension diagnosis in youth
        chol_threshold = 230
    else:  # Adult population
        systolic_bp = np.random.normal(130, 10, n)  # Higher baseline BP for adults
        cholesterol = np.random.normal(200, 25, n)
        bp_threshold = 140  # Standard adult threshold for hypertension
        chol_threshold = 240

    # Create DataFrame with medical features
    data = pd.DataFrame({
        'age': np.random.randint(age_range[0], age_range[1], n),
        # BMI tends to be lower in youth compared to adults
        'bmi': np.random.normal(25 if age_range[0] > 20 else 20, 3, n),
        'systolic_bp': systolic_bp,
        'cholesterol': cholesterol,
        'smoker': np.random.randint(0, 2, n),  # Binary feature: 0=non-smoker, 1=smoker
    })

    # Create target variable: hypertension diagnosis based on BP or cholesterol
    data['hypertension'] = ((systolic_bp > bp_threshold) | 
                           (cholesterol > chol_threshold)).astype(int)
    return data

# Generate our two datasets
adult_data = simulate_data(n_adults, (30, 65))  # Source domain: adults
youth_data = simulate_data(n_youth, (13, 20))   # Target domain: youth

# =====================================================================
# PREPARE ADULT DATA (SOURCE DOMAIN)
# =====================================================================

# Split features (X) and target (y)
X_adult = adult_data.drop('hypertension', axis=1)
y_adult = adult_data['hypertension']

# Split into training and validation sets (80% train, 20% validation)
X_train_adult, X_val_adult, y_train_adult, y_val_adult = train_test_split(
    X_adult, y_adult, test_size=0.2, random_state=42
)

# Standardize features to have zero mean and unit variance
# This is important for neural networks to converge properly
scaler = StandardScaler()
X_train_adult_scaled = scaler.fit_transform(X_train_adult)
X_val_adult_scaled = scaler.transform(X_val_adult)  # Use same scaler for validation set

# =====================================================================
# MODEL DEFINITION
# =====================================================================

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron with a body (feature extractor) and head (classifier).

    This architecture is designed to demonstrate transfer learning by separating:
    - The body: multiple layers that learn feature representations
    - The head: final layer that makes predictions based on learned features

    Args:
        input_dim (int): Number of input features
    """
    def __init__(self, input_dim):
        super().__init__()
        # The body extracts features from the input data
        # In transfer learning, we often freeze these layers after initial training
        self.body = nn.Sequential(
            nn.Linear(input_dim, 32),  # First hidden layer: input_dim → 32 neurons
            nn.ReLU(),                 # Non-linear activation function
            nn.Linear(32, 16),         # Second hidden layer: 32 → 16 neurons
            nn.ReLU()                  # Another non-linearity
        )

        # The head makes predictions based on the extracted features
        # In transfer learning, we often only fine-tune this layer
        self.head = nn.Linear(16, 1)   # Output layer: 16 → 1 (binary classification)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            torch.Tensor: Predicted probabilities of shape [batch_size, 1]
        """
        # Pass input through body to extract features
        x = self.body(x)
        # Pass features through head and apply sigmoid for binary classification
        return torch.sigmoid(self.head(x))  # Sigmoid ensures output is between 0 and 1

def to_tensor(X, y):
    """
    Convert numpy arrays or pandas DataFrames to PyTorch tensors.

    Args:
        X (numpy.ndarray): Feature matrix
        y (pandas.Series): Target variable

    Returns:
        tuple: (X_tensor, y_tensor) as PyTorch tensors with appropriate types
    """
    return (torch.tensor(X, dtype=torch.float32),  # Features as float32
            torch.tensor(y.values, dtype=torch.float32).view(-1, 1))  # Target as column vector

# Convert our data to PyTorch tensors
X_train_adult_tensor, y_train_adult_tensor = to_tensor(X_train_adult_scaled, y_train_adult)
X_val_adult_tensor, y_val_adult_tensor = to_tensor(X_val_adult_scaled, y_val_adult)

# =====================================================================
# MODEL TRAINING (SOURCE DOMAIN)
# =====================================================================

# Initialize model, loss function, and optimizer
model = MLP(input_dim=X_train_adult_scaled.shape[1])
criterion = nn.BCELoss()  # Binary Cross Entropy loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer with learning rate 0.01

def train_model(model, X, y, epochs=50):
    """
    Train a PyTorch model.

    Args:
        model (nn.Module): PyTorch model to train
        X (torch.Tensor): Input features
        y (torch.Tensor): Target values
        epochs (int): Number of training epochs
    """
    for epoch in range(epochs):
        # Set model to training mode (affects dropout, batch norm, etc.)
        model.train()

        # Zero gradients from previous step
        optimizer.zero_grad()

        # Forward pass: compute predictions
        y_pred = model(X)

        # Compute loss
        loss = criterion(y_pred, y)

        # Backward pass: compute gradients
        loss.backward()

        # Update weights based on gradients
        optimizer.step()

# Train the model on adult data
print("Training model on adult data...")
train_model(model, X_train_adult_tensor, y_train_adult_tensor)

# =====================================================================
# EVALUATE ON ADULT DATA (SOURCE DOMAIN)
# =====================================================================

# Set model to evaluation mode (affects dropout, batch norm, etc.)
model.eval()

# Disable gradient calculation for inference (saves memory and computation)
with torch.no_grad():
    # Get predictions on validation set
    val_preds_adult = model(X_val_adult_tensor).round()  # Round to get binary predictions

    # Calculate performance metrics
    adult_precision, adult_recall, adult_f1, _ = precision_recall_fscore_support(
        y_val_adult_tensor, val_preds_adult, average='binary'
    )

print("\n--- Adult Validation Performance ---")
print(f"Precision: {adult_precision:.4f}")
print(f"Recall:    {adult_recall:.4f}")
print(f"F1 Score:  {adult_f1:.4f}")

# =====================================================================
# PREPARE YOUTH DATA (TARGET DOMAIN)
# =====================================================================

# Split features and target for youth data
X_youth = youth_data.drop('hypertension', axis=1)
y_youth = youth_data['hypertension']

# Split into training and validation sets
X_train_youth, X_val_youth, y_train_youth, y_val_youth = train_test_split(
    X_youth, y_youth, test_size=0.2, random_state=42
)

# Use the SAME scaler that was fit on adult data
# This ensures consistent feature scaling between domains
X_train_youth_scaled = scaler.transform(X_train_youth)
X_val_youth_scaled = scaler.transform(X_val_youth)

# Convert to PyTorch tensors
X_train_youth_tensor, y_train_youth_tensor = to_tensor(X_train_youth_scaled, y_train_youth)
X_val_youth_tensor, y_val_youth_tensor = to_tensor(X_val_youth_scaled, y_val_youth)

# =====================================================================
# EVALUATE ON YOUTH DATA BEFORE FINE-TUNING
# =====================================================================

# Evaluate the adult-trained model on youth data (before fine-tuning)
model.eval()
with torch.no_grad():
    val_preds_youth_before = model(X_val_youth_tensor).round()
    youth_precision_before, youth_recall_before, youth_f1_before, _ = precision_recall_fscore_support(
        y_val_youth_tensor, val_preds_youth_before, average='binary'
    )

print("\n--- Youth Validation Performance (Before Finetuning) ---")
print(f"Precision: {youth_precision_before:.4f}")
print(f"Recall:    {youth_recall_before:.4f}")
print(f"F1 Score:  {youth_f1_before:.4f}")

# =====================================================================
# TRANSFER LEARNING: FREEZE BODY, FINE-TUNE HEAD
# =====================================================================

# Display current trainable parameters
print("\n--- Trainable Parameters Before Freezing ---")
for name, param in model.named_parameters():
    print(f"{name:30} - Trainable: {param.requires_grad}")

# Freeze the body (feature extractor) parameters
# This is the key step in transfer learning - we keep the learned features
# but allow the classifier to adapt to the new domain
print("\n--- Freezing model body (feature extractor) ---")
for param in model.body.parameters():
    param.requires_grad = False  # Freeze parameters (no gradient updates)

# Reinitialize the head (classifier) for the new domain
# This allows the model to learn new decision boundaries for youth data
model.head = nn.Linear(16, 1)  # Same architecture but fresh weights

# Create a new optimizer that only updates the head parameters
# Note: Only parameters with requires_grad=True will be updated
optimizer = optim.Adam(model.head.parameters(), lr=0.01)

# Display updated trainable parameters
print("\n--- Updated trainable parameters after freezing ---")
for name, param in model.named_parameters():
    print(f"{name:30} - Trainable: {param.requires_grad}")

# Fine-tune the model on youth data (with fewer epochs)
print("\nFine-tuning model on youth data...")
train_model(model, X_train_youth_tensor, y_train_youth_tensor, epochs=20)

# =====================================================================
# EVALUATE ON YOUTH DATA AFTER FINE-TUNING
# =====================================================================

# Evaluate the fine-tuned model on youth validation data
model.eval()
with torch.no_grad():
    val_preds_youth = model(X_val_youth_tensor).round()
    youth_precision, youth_recall, youth_f1, _ = precision_recall_fscore_support(
        y_val_youth_tensor, val_preds_youth, average='binary'
    )

print("\n--- Youth Validation Performance (After Finetuning) ---")
print(f"Precision: {youth_precision:.4f}")
print(f"Recall:    {youth_recall:.4f}")
print(f"F1 Score:  {youth_f1:.4f}")

# =====================================================================
# RESULTS SUMMARY
# =====================================================================

# Create a summary table comparing performance across domains
print("\n--- Performance Comparison Summary ---")
report_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-score'],
    'Adult': [adult_precision, adult_recall, adult_f1],
    'Youth (Before Fine-tuning)': [youth_precision_before, youth_recall_before, youth_f1_before],
    'Youth (After Fine-tuning)': [youth_precision, youth_recall, youth_f1]
})
print(report_df)
