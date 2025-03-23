import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
# Import the normalization script for preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Preprocessing import normalization_small

# Set manual seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def augment_time_series(x, noise_level=0.01):
    """
    Apply data augmentation to time series data.
    Adds Gaussian noise and applies a simple random time warping.
    
    Args:
        x (Tensor): Input tensor of shape [batch_size, seq_len, features].
        noise_level (float): Standard deviation of Gaussian noise.
        
    Returns:
        Tensor: Augmented tensor.
    """
    # Add Gaussian noise
    noise = torch.randn_like(x) * noise_level
    x = x + noise
    
    # Random time warping (simple implementation)
    if random.random() > 0.5:
        scale = random.uniform(0.9, 1.1)
        x = F.interpolate(
            x.transpose(1, 2), 
            scale_factor=scale, 
            mode='linear', 
            align_corners=False
        ).transpose(1, 2)
        
        # Ensure output length matches input length
        if x.size(1) > 60:
            x = x[:, :60, :]
        elif x.size(1) < 60:
            x = F.pad(x, (0, 0, 0, 60 - x.size(1)))
    
    return x


# =============================================================================
# Preprocessing and Data Loading Functions
# =============================================================================
def preprocess_data(dataset_path, save_directory, norm=False, num_samples=288):
    """
    Preprocess and save the dataset using the provided normalization script.
    This function will create train, test, and validation splits.
    """
    normalization_small.normalize_balanced_dataset(
        dataset_path=dataset_path,
        save_directory=save_directory,
        norm=norm,
        num_samples=num_samples
    )
    print("Data preprocessing complete.\n")



def load_dataloaders(data_path, batch_size=32):
    """
    Load the preprocessed numpy data (train, validation, test) and return PyTorch DataLoaders.
    For transformers, we keep the data in [batch, seq_len, features] format.
    """
    # Load datasets
    X_train = np.load(os.path.join(data_path, "train.npy"))
    y_train = np.load(os.path.join(data_path, "train_labels.npy"))
    X_val   = np.load(os.path.join(data_path, "val.npy"))
    y_val   = np.load(os.path.join(data_path, "val_labels.npy"))
    X_test  = np.load(os.path.join(data_path, "test.npy"))
    y_test  = np.load(os.path.join(data_path, "test_labels.npy"))

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)  # [batch, seq, features]
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)

    # Squeeze labels to get shape [batch]
    y_train = torch.tensor(y_train, dtype=torch.float32).squeeze()
    y_val   = torch.tensor(y_val, dtype=torch.float32).squeeze()
    y_test  = torch.tensor(y_test, dtype=torch.float32).squeeze()

    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    return train_loader, val_loader, test_loader


# =============================================================================
# Positional Encoding for Transformer
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============================================================================
# Transformer-based Model for Ransomware Detection
# =============================================================================
class TransformerRansomwareDetector(nn.Module):
    def __init__(self, num_features=23, seq_len=60, d_model=32, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout,
            batch_first=True  
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, x, attention_mask=None):
        # x: [batch, seq_len, num_features]
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        x = self.pos_encoder(x)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (x.sum(dim=-1) != 0)  # [batch, seq_len]
        
        # Apply transformer with residual connection
        transformer_out = self.transformer_encoder(x, src_key_padding_mask=~attention_mask)
        x = x + transformer_out  # Residual connection
        
        # Improved pooling
        x = x.transpose(1, 2)  # [batch, d_model, seq_len]
        x = self.pool(x).squeeze(-1)  # [batch, d_model]
        x = self.fc(x)
        return torch.sigmoid(x).squeeze()


def build_model(train_loader, epochs):
    """
    Build the transformer model, optimizer, loss criterion, and scheduler.
    Here we use AdamW with weight decay and a warmup scheduler from transformers.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerRansomwareDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler setup
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = num_training_steps // 10  # 10% of total steps for warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return model, criterion, optimizer, scheduler, device


# =============================================================================
# Training Function
# =============================================================================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=150, patience=15, save_dir=r"../Models/transformer_ransomware_model"):
    """
    Train the model with early stopping, gradient clipping, and learning rate scheduling.
    Applies data augmentation during training.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for inputs, labels in train_loader:
            # Apply data augmentation during training
            inputs = augment_time_series(inputs)
            
            # Generate attention mask
            attention_mask = (inputs.sum(dim=-1) != 0).to(device)
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                attention_mask = (inputs.sum(dim=-1) != 0).to(device)
                outputs = model(inputs, attention_mask)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, os.path.join(save_dir, "best_model2.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    return train_losses, val_losses


# =============================================================================
# Evaluation Function
# =============================================================================
def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model on the test set.
    Computes metrics (accuracy, precision, recall, F1-score) and plots a confusion matrix.
    """

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).cpu().numpy()  # Binary predictions
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Calculate evaluation metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("Evaluation Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Safe (0)", "Malware (1)"], yticklabels=["Safe (0)", "Malware (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def plot_loss_curves(train_losses, val_losses):
    """
    Plot the training and validation loss curves.
    """

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    # Configuration parameters.
    config = {
        'dataset_path': r"../Dataset",
        'preprocess_save_dir': r"../Transformer",
        'model_save_dir': r"../Models/transformer_ransomware_model",
        'batch_size': 32,
        'num_samples': 288,
        'norm': False,  # Use MinMax normalization
        'epochs': 150,
        'patience': 15
    }
    
    # Preprocess the data using the provided normalization script.
    preprocess_data(config['dataset_path'], config['preprocess_save_dir'], 
                    norm=config['norm'], num_samples=config['num_samples'])
    
    # Load DataLoaders for train, validation, and test sets.
    train_loader, val_loader, test_loader = load_dataloaders(
        config['preprocess_save_dir'], batch_size=config['batch_size']
    )
    
    # Build the model along with loss, optimizer, and scheduler.
    model, criterion, optimizer, scheduler, device = build_model(train_loader, config['epochs'])
    
    # Train the model with early stopping and gradient clipping.
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device,
        epochs=config['epochs'], patience=config['patience'], save_dir=config['model_save_dir']
    )
    
    # Plot training and validation loss curves.
    plot_loss_curves(train_losses, val_losses)
    
    # Evaluate the model on the test set with metrics and confusion matrix.
    evaluate_model(model, test_loader, device)

if __name__ == "__main__":
    main()
    