import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split
from sklearn.model_selection import KFold, train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
# Import the normalization file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Preprocessing import norm_small_noval

# Set manual seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =============================================================================
# LSTM Model Definition
# =============================================================================
class LSTMRansomwareDetector(nn.Module):
    def __init__(self, input_size=23, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMRansomwareDetector, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (hn, cn) = self.lstm(x)
        # Use the hidden state from the last LSTM layer (last time step)
        last_hidden = hn[-1]
        out = self.fc(last_hidden)
        return torch.sigmoid(out).squeeze()

# =============================================================================
# Data Loading Function
# =============================================================================
def load_data(DATA_PATH=r"../LSTM"):
    """Load normalized data from numpy files and return train and test tensors."""
    X_train = np.load(os.path.join(DATA_PATH, "train.npy"))
    y_train = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "test_labels.npy"))
    
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).squeeze()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).squeeze()
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# =============================================================================
# K-Fold Cross Validation Training Function
# =============================================================================
def train_kfold(MODEL_SAVE_DIR, model_class, dataset, num_epochs=100, batch_size=32, learning_rate=0.001, patience=10, k_folds=5, device="cpu"):
    
    # Performs k-fold cross validation training and saves the best model per fold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=SEED)
    fold_results = {}

    print("Starting k-fold cross validation...")
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        model = model_class().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
            train_loss /= len(train_subset)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
            val_loss /= len(val_subset)
            
            print(f"Fold {fold+1} - Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_path = os.path.join(MODEL_SAVE_DIR, f"lstm_fold{fold+1}_best.pth")
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1} for fold {fold+1}")
                    break
        fold_results[fold] = best_val_loss
        print(f"Fold {fold+1} best validation loss: {best_val_loss:.4f}")
    
    avg_val_loss = sum(fold_results.values()) / k_folds
    print(f"\nAverage Validation Loss across {k_folds} folds: {avg_val_loss:.4f}")
    return fold_results

# =============================================================================
# Final Training with Validation Split (for Loss Curve Plotting)
# =============================================================================
def train_final(MODEL_SAVE_DIR, model_class, X_train, y_train, num_epochs=100, batch_size=32, learning_rate=0.001, patience=10, device="cpu"):
    """Train the model with an internal train/validation split, record loss curves, and save the best model."""

    # Split training data (80% train, 20% validation)
    train_tensor, val_tensor, train_labels, val_labels = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED)
    
    train_dataset = TensorDataset(train_tensor, train_labels)
    val_dataset = TensorDataset(val_tensor, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = model_class().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print("\nTraining final model (with validation split) for loss curve plotting...")
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)
        epoch_train_loss /= len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation 
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)
        epoch_val_loss /= len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Final Model - Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f} - Val Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            best_model_path = os.path.join(MODEL_SAVE_DIR, "lstm_final_model.pth")
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered for final model at epoch {epoch+1}")
                break

    return model, train_losses, val_losses

# =============================================================================
# Evaluation Function
# =============================================================================
def evaluate_model(model, X_test, y_test, batch_size=32, device="cpu"):
    """Evaluate the model on the test set and print the confusion matrix and classification report."""

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_preds = []
    all_labels = []
    criterion = nn.BCELoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    # Generate confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Safe", "Malware"], digits=4)
    
    # Print results in the desired format
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return all_labels, all_preds

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_loss_curves(train_losses, val_losses):
    """Plot training and validation loss curves."""

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss", marker="o")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(all_labels, all_preds):
    """Plot the confusion matrix."""

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe (0)", "Malware (1)"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix on Test Set")
    plt.show()

# =============================================================================
# Main Function
# =============================================================================
def main():

    # --- Data Directories ---
    # Set the dataset directory and the directory where normalized data are saved.
    dataset_path = r"../Dataset"   
    data_directory = r"../LSTM"      
    model_save_directory = r"../Models/lstm_ransomware_model"  
    
    # This function loads the CSV files, normalizes them, and saves them as numpy files
    norm_small_noval.normalize_balanced_dataset(dataset_path=dataset_path, save_directory=data_directory, norm=False, test_ratio=0.2, num_samples=288)

    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_data(data_directory)
    
    # Create dataset for k-fold CV
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    # K-Fold Cross Validation Training
    print("====== K-Fold Cross Validation Training ======")
    _ = train_kfold(model_save_directory, LSTMRansomwareDetector, train_dataset, num_epochs=100, batch_size=64, 
                    learning_rate=0.001, patience=10, k_folds=10, device=device)
    
    # Final Training with a Validation Split for Loss Curve Plotting
    print("\n====== Final Training (with Validation Split) ======")
    final_model, train_losses, val_losses = train_final(model_save_directory, LSTMRansomwareDetector, 
                                                        X_train_tensor, y_train_tensor, 
                                                        num_epochs=100, batch_size=64, 
                                                        learning_rate=0.001, patience=10, device=device)
    plot_loss_curves(train_losses, val_losses)
    
    # Evaluate the Final Model on the Test Set and Plot Confusion Matrix
    print("\n====== Evaluation on Test Set ======")
    # Load best final model
    final_model.load_state_dict(torch.load(os.path.join(model_save_directory, "lstm_final_model.pth")))
    all_labels, all_preds = evaluate_model(final_model, X_test_tensor, y_test_tensor, batch_size=64, device=device)
    plot_confusion_matrix(all_labels, all_preds)

if __name__ == "__main__":
    main()

