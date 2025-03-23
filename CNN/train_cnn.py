import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Preprocessing import normalization_small

# Set manual seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Load and Normalize the dataset
normalization_small.normalize_balanced_dataset(dataset_path=r"../Dataset", save_directory=r"../CNN", norm=False, num_samples=288)

# Load Preprocessed Data
DATA_PATH = r"../CNN"
X_train = np.load(os.path.join(DATA_PATH, "train.npy"))
y_train = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
X_val = np.load(os.path.join(DATA_PATH, "val.npy"))
y_val = np.load(os.path.join(DATA_PATH, "val_labels.npy"))

# Reshape for Conv1D
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Create DataLoader with larger batch size
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

# Define CNN Model
class CNNRansomwareDetector(nn.Module):
    def __init__(self):
        super(CNNRansomwareDetector, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=23, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.flatten = nn.Flatten()
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 23, 60)
            dummy_output = self.forward_conv(dummy_input)
            self.fc_input_size = dummy_output.shape[1]
        
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.batch_norm_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)

        # Droupout layers with different rates
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        
    def forward_conv(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.flatten(x)
        return x
        
    def forward(self, x):
        x = self.forward_conv(x)
        
        # Three fully connected layers with dropout
        x = self.fc1(x)
        x = self.batch_norm_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.batch_norm_fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze() # Squeeze to match the shape of the labels

# Initialize model, criterion, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNRansomwareDetector().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Add learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    factor=0.5, 
    patience=7,
    min_lr=1e-6)

# Training Loop with validation
epochs = 150
best_val_loss = float('inf')
patience = 15
patience_counter = 0

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            val_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    print(f"Epoch {epoch+1}, Current Learning Rate: {scheduler.optimizer.param_groups[0]['lr']}")
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), os.path.join(r"../Models/cnn_ransomware_model", "cnn_model.pth"))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Plot Loss Curves
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curves")
plt.legend()
plt.grid()
plt.show()


