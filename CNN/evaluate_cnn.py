import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define paths
DATA_PATH = r"../CNN"  # Preprocessed test data
MODEL_DIR = r"../Models/cnn_ransomware_model"
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")

# Load Preprocessed Test Data
X_test = np.load(os.path.join(DATA_PATH, "test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "test_labels.npy"))

# Reshape input for Conv1D
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
y_test = torch.tensor(y_test, dtype=torch.float32)

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
        
        # Three fully connected layers with batch normalization
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.batch_norm_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch_norm_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        
        # Dropout layers with adjusted rates
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
        
        x = self.fc1(x)
        x = self.batch_norm_fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.batch_norm_fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNRansomwareDetector().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Model loaded from {MODEL_PATH}")

# Create DataLoader for evaluation
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test, y_test), batch_size=32)

# Track test loss
criterion = nn.BCELoss()
test_losses = []
y_true = []
y_pred = []

# Run Evaluation
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels.squeeze())
        test_losses.append(loss.item())
        
        # Store predictions and labels
        y_true.extend(labels.cpu().numpy())
        y_pred.extend((outputs.cpu().numpy() > 0.5).astype(int))

# Compute Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification Report
print("\n🔹 Classification Report:\n", classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Safe", "Malware"], 
            yticklabels=["Safe", "Malware"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Plot Test Loss Curve
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(test_losses) + 1), test_losses, 
         marker="o", linestyle="-", color="red")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Test Loss Curve")
plt.grid()
plt.show()
