import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support

# Simulate a medical dataset for adults and young people
np.random.seed(42)
n_adults = 1000
n_youth = 300

def simulate_data(n, age_range):
    data = pd.DataFrame({
        'age': np.random.randint(age_range[0], age_range[1], n),
        'bmi': np.random.normal(25 if age_range[0] > 20 else 20, 3, n),
        'systolic_bp': np.random.normal(130 if age_range[0] > 20 else 115, 10, n),
        'cholesterol': np.random.normal(200, 25, n),
        'smoker': np.random.randint(0, 2, n),
    })
    # Target: Hypertension if bp > 140 or cholesterol > 240
    data['hypertension'] = ((data['systolic_bp'] > 140) | (data['cholesterol'] > 240)).astype(int)
    return data

adult_data = simulate_data(n_adults, (30, 65))
youth_data = simulate_data(n_youth, (13, 20))

# Prepare adult data
X_adult = adult_data.drop('hypertension', axis=1)
y_adult = adult_data['hypertension']
X_train_adult, X_val_adult, y_train_adult, y_val_adult = train_test_split(X_adult, y_adult, test_size=0.2, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_adult_scaled = scaler.fit_transform(X_train_adult)
X_val_adult_scaled = scaler.transform(X_val_adult)



# Define a simple MLP
class MLP(nn.Module):
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

# Convert data to torch tensors
def to_tensor(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

X_train_adult_tensor, y_train_adult_tensor = to_tensor(X_train_adult_scaled, y_train_adult)
X_val_adult_tensor, y_val_adult_tensor = to_tensor(X_val_adult_scaled, y_val_adult)

# Pretrain model on adult data
model = MLP(input_dim=X_train_adult_scaled.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_model(model, X, y, epochs=50):
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

train_model(model, X_train_adult_tensor, y_train_adult_tensor)

# Evaluate on adult validation set
model.eval()
with torch.no_grad():
    val_preds_adult = model(X_val_adult_tensor).round()
    adult_precision, adult_recall, adult_f1, _ = precision_recall_fscore_support(
        y_val_adult_tensor, val_preds_adult, average='binary'
    )

# Finetune on youth data (only head layer)
X_youth = youth_data.drop('hypertension', axis=1)
y_youth = youth_data['hypertension']
X_train_youth, X_val_youth, y_train_youth, y_val_youth = train_test_split(X_youth, y_youth, test_size=0.2, random_state=42)
X_train_youth_scaled = scaler.transform(X_train_youth)
X_val_youth_scaled = scaler.transform(X_val_youth)
X_train_youth_tensor, y_train_youth_tensor = to_tensor(X_train_youth_scaled, y_train_youth)
X_val_youth_tensor, y_val_youth_tensor = to_tensor(X_val_youth_scaled, y_val_youth)

# Freeze body, reinit head
for param in model.body.parameters():
    param.requires_grad = False
model.head = nn.Linear(16, 1)
optimizer = optim.Adam(model.head.parameters(), lr=0.01)

train_model(model, X_train_youth_tensor, y_train_youth_tensor, epochs=20)

# Evaluate on youth validation
model.eval()
with torch.no_grad():
    val_preds_youth = model(X_val_youth_tensor).round()
    youth_precision, youth_recall, youth_f1, _ = precision_recall_fscore_support(
        y_val_youth_tensor, val_preds_youth, average='binary'
    )

# Display reports
report_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-score'],
    'Adult': [adult_precision, adult_recall, adult_f1],
    'Youth (Finetuned)': [youth_precision, youth_recall, youth_f1]
})
