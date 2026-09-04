import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import joblib

from src.models.housing import HousingMLP

# Load dataset
df = pd.read_csv('housing.csv')

# Drop unused column
df = df.drop(columns=['ocean_proximity'])

# Separate features and target
X = df.drop(columns=['median_house_value']).values
y = df['median_house_value'].values.reshape(-1, 1)

# Train/validation split (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Impute missing values using training set median
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

# Feature scaling
scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_val = scaler_x.transform(X_val)

# Target scaling
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create PyTorch datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


input_dim = X_train.shape[1]
model = HousingMLP(input_dim)

# Custom RMSE Loss Function
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)

criterion = RMSELoss()

lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 2000
patience = 10
best_val_loss = float('inf')
patience_counter = 0
best_model_weights = None

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            val_loss += loss.item() * batch_X.size(0)

            val_preds.append(predictions.numpy())
            val_targets.append(batch_y.numpy())

    val_loss /= len(val_loader.dataset)

    # Concatenate predictions and unscale back to original units
    val_preds = np.vstack(val_preds)
    val_targets = np.vstack(val_targets)
    val_preds_orig = scaler_y.inverse_transform(val_preds)
    val_targets_orig = scaler_y.inverse_transform(val_targets)

    # Compute evaluation metrics (R2 and MAPE)
    r2 = r2_score(val_targets_orig, val_preds_orig)
    mape = mean_absolute_percentage_error(val_targets_orig, val_preds_orig) * 100

    print(f"Epoch {epoch+1:03d} | Train Loss (scaled): {train_loss:.4f} | "
          f"Val Loss (scaled): {val_loss:.4f} | R2 Score: {r2:.4f} | MAPE: {mape:.2f}%")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_weights = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

# Restore best model weights
if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

# Create output directory if it does not exist
output_dir = 'artifacts'
os.makedirs(output_dir, exist_ok=True)

# Define file paths
model_path = os.path.join(output_dir, 'housing_model.pt')
imputer_path = os.path.join(output_dir, 'imputer.joblib')
scaler_x_path = os.path.join(output_dir, 'scaler_x.joblib')
scaler_y_path = os.path.join(output_dir, 'scaler_y.joblib')
metadata_path = os.path.join(output_dir, 'metadata.json')

# Save the complete PyTorch model, including its architecture and weights
torch.save(model, model_path)

# Save preprocessors using joblib
joblib.dump(imputer, imputer_path)
joblib.dump(scaler_x, scaler_x_path)
joblib.dump(scaler_y, scaler_y_path)

# Construct metadata JSON
metadata = {
    "timestamp": datetime.now().isoformat(),
    "model_architecture": {
        "type": "HousingMLP",
        "input_dim": input_dim,
        "layers": [
            {"type": "Linear", "in_features": input_dim, "out_features": 32},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 32, "out_features": 32},
            {"type": "ReLU"},
            {"type": "Linear", "in_features": 32, "out_features": 1}
        ]
    },
    "preprocessing": {
        "imputer_strategy": "median",
        "scaler_x": "StandardScaler",
        "scaler_y": "StandardScaler"
    },
    "hyperparameters": {
        "batch_size": batch_size,
        "learning_rate": lr,
        "optimizer": "Adam",
        "loss_function": "RMSELoss",
        "patience": patience,
        "max_epochs": epochs,
        "stopped_epoch": epoch + 1
    },
    "metrics": {
        "best_val_loss": float(best_val_loss),
        "final_r2_score": float(r2),
        "final_mape_percent": float(mape)
    },
    "artifacts_directory": output_dir,
    "artifacts": {
        "model": "housing_model.pt",
        "imputer": "imputer.joblib",
        "scaler_x": "scaler_x.joblib",
        "scaler_y": "scaler_y.joblib"
    }
}

# Save metadata to JSON file inside artifacts folder
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)