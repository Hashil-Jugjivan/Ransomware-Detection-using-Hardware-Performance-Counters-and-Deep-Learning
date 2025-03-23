import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Set paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Preprocessing import norm_small_noval

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_PATH = r"../LSTM_NAS"
MODELS_PATH = r"../Models/nas_lstm_ransomware_model"
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# Set manual seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Check if data exists, if not preprocess it
if not os.path.exists(os.path.join(DATA_PATH, "train.npy")):
    logger.info("Preprocessing data...")
    norm_small_noval.normalize_balanced_dataset(
        dataset_path=r"../Dataset", 
        save_directory=DATA_PATH, 
        norm=False, 
        test_ratio=0.2,
        num_samples=288
    )

# =============================================================================
# Data Loading Function
# =============================================================================
def load_data(batch_size=32):
    """Load normalized data from numpy files and create DataLoaders."""
    X_train = np.load(os.path.join(DATA_PATH, "train.npy"))
    y_train = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "test_labels.npy"))
    
    # Split training data into train and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(X_train))
    val_size = len(X_train) - train_size
    
    X_val = X_train[train_size:]
    y_val = y_train[train_size:]
    X_train = X_train[:train_size]
    y_train = y_train[:train_size]
    
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).squeeze()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).squeeze()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).squeeze()
    
    # Print shapes for verification
    logger.info(f"Data shapes: X_train={X_train_tensor.shape}, X_val={X_val_tensor.shape}, X_test={X_test_tensor.shape}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# =============================================================================
# Dynamic LSTM Model Definition
# =============================================================================
class DynamicLSTMRansomwareDetector(nn.Module):
    def __init__(self, 
                 input_size=23, 
                 hidden_sizes=[64, 32], 
                 num_layers=2, 
                 dropout=0.3,
                 bidirectional=False,
                 fc_layers=[64, 32],
                 fc_dropout=0.3,
                 use_batch_norm=True,
                 activation='relu'):
        
        super(DynamicLSTMRansomwareDetector, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.activation_name = activation
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Tanh()
        
        # Create LSTM layers
        self.lstm_layers = nn.ModuleList()
        lstm_input_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # If this is the last LSTM layer, we need to consider return_sequences
            return_sequences = i < len(hidden_sizes) - 1
            
            lstm = nn.LSTM(
                input_size=lstm_input_size,
                hidden_size=hidden_size,
                num_layers=1,  # Use single layer LSTM and stack them manually
                batch_first=True,
                bidirectional=bidirectional,
                dropout=0 if i == len(hidden_sizes) - 1 else dropout
            )
            
            self.lstm_layers.append(lstm)
            lstm_input_size = hidden_size * self.num_directions
        
        # Create batch norm layers for LSTM outputs
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList()
            for hidden_size in hidden_sizes[:-1]:  # No batch norm after last LSTM
                self.batch_norm_layers.append(nn.BatchNorm1d(hidden_size * self.num_directions))
        
        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_batch_norm_layers = nn.ModuleList()
        self.fc_dropout_layers = nn.ModuleList()
        
        # Input size for first FC layer is the last LSTM hidden size
        fc_input_size = hidden_sizes[-1] * self.num_directions
        
        for i, fc_size in enumerate(fc_layers):
            self.fc_layers.append(nn.Linear(fc_input_size, fc_size))
            if use_batch_norm:
                self.fc_batch_norm_layers.append(nn.BatchNorm1d(fc_size))
            self.fc_dropout_layers.append(nn.Dropout(fc_dropout))
            fc_input_size = fc_size
        
        # Output layer
        self.fc_output = nn.Linear(fc_input_size, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Process through LSTM layers
        for i, lstm in enumerate(self.lstm_layers):
            # For intermediate layers, need all time steps
            # For the last layer, only need the last time step
            is_last_layer = i == len(self.lstm_layers) - 1
            
            output, (h_n, c_n) = lstm(x)
            
            if is_last_layer:
                # For the last layer, use the final hidden state
                if self.bidirectional:
                    # Concatenate forward and backward directions
                    x = torch.cat((h_n[0], h_n[1]), dim=1)
                else:
                    x = h_n[0]  # Shape: [batch_size, hidden_size]
            else:
                # For intermediate layers, use all time steps for next layer
                x = output  # Shape: [batch_size, seq_len, hidden_size * num_directions]
                
                # Apply batch normalization if enabled
                if self.use_batch_norm:
                    # Need to reshape for BatchNorm1d
                    orig_shape = x.shape
                    x = x.reshape(-1, orig_shape[2])  # [batch_size * seq_len, hidden_size * num_directions]
                    x = self.batch_norm_layers[i](x)
                    x = x.reshape(orig_shape)  # Back to [batch_size, seq_len, hidden_size * num_directions]
                
                x = self.activation(x)
        
        # Process through fully connected layers
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if self.use_batch_norm:
                x = self.fc_batch_norm_layers[i](x)
            x = self.activation(x)
            x = self.fc_dropout_layers[i](x)
        
        # Final output layer
        x = torch.sigmoid(self.fc_output(x))
        return x.squeeze()

# =============================================================================
# Training Functions
# =============================================================================
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, patience, epochs=100, model_path=None):
    """Train and validate the model with early stopping."""

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model 
            if model_path:
                torch.save(model.state_dict(), model_path)
                logger.info(f"Model saved to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Log progress every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs}, "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}, "
                        f"LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}")
    
    return best_val_loss, train_losses, val_losses

# =============================================================================
# Evaluation Function
# =============================================================================
def evaluate_model(model, test_loader):
    """Evaluate the model on the test set and calculate performance metrics."""

    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    test_loss = 0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            
            # Convert outputs to binary predictions
            preds = (outputs > 0.5).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(outputs.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Generate confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Safe", "Malware"])
    
    results = {
        'loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_scores': y_scores
    }
    
    return results

# =============================================================================
# Plotting Functions
# =============================================================================
def plot_loss_curves(train_losses, val_losses, save_path):
    """Plot training and validation loss curves."""

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Training Loss", marker="o", markersize=3)
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def plot_confusion_matrix(cm, save_path):
    """Plot the confusion matrix."""

    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Safe", "Malware"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# =============================================================================
# Neural Architecture Search with Optuna
# =============================================================================
def objective(trial):
    """Objective function for Optuna to optimize model architecture and hyperparameters."""

    try:
        # Sample hyperparameters
        
        # LSTM architecture parameters
        num_lstm_layers = trial.suggest_int("num_lstm_layers", 1, 3)
        lstm_hidden_sizes = []
        
        # Generate a list of decreasing hidden sizes
        first_hidden_size = trial.suggest_categorical("first_hidden_size", [32, 64, 128, 256])
        lstm_hidden_sizes.append(first_hidden_size)
        
        for i in range(1, num_lstm_layers):
            # Each subsequent layer has fewer units
            prev_size = lstm_hidden_sizes[i-1]
            size_options = [s for s in [16, 32, 64, 128] if s <= prev_size]
            if not size_options:  # Fallback if no valid options
                size_options = [16]
            hidden_size = trial.suggest_categorical(f"lstm_hidden_size_{i}", size_options)
            lstm_hidden_sizes.append(hidden_size)
        
        lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.5)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])
        
        # Fully connected layers
        num_fc_layers = trial.suggest_int("num_fc_layers", 0, 2)
        fc_layers = []
        
        for i in range(num_fc_layers):
            fc_size = trial.suggest_categorical(f"fc_size_{i}", [16, 32, 64, 128])
            fc_layers.append(fc_size)
        
        fc_dropout = trial.suggest_float("fc_dropout", 0.1, 0.5)
        
        # Other hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
        activation = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "tanh"])
        patience = trial.suggest_int("patience", 10, 20)
        
        # Print the sampled parameters for this trial
        logger.info(f"Trial {trial.number} - Hyperparameters:")
        logger.info(f"  LSTM Layers: {num_lstm_layers}, Hidden Sizes: {lstm_hidden_sizes}")
        logger.info(f"  Bidirectional: {bidirectional}, Dropout: {lstm_dropout}")
        logger.info(f"  FC Layers: {fc_layers}, FC Dropout: {fc_dropout}")
        logger.info(f"  Batch Size: {batch_size}, Learning Rate: {learning_rate}")
        logger.info(f"  Weight Decay: {weight_decay}, Batch Norm: {use_batch_norm}")
        logger.info(f"  Activation: {activation}, Patience: {patience}")
        
        # Load data
        train_loader, val_loader, _ = load_data(batch_size)
        
        # Create model
        model = DynamicLSTMRansomwareDetector(
            input_size=23,
            hidden_sizes=lstm_hidden_sizes,
            num_layers=num_lstm_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            fc_layers=fc_layers,
            fc_dropout=fc_dropout,
            use_batch_norm=use_batch_norm,
            activation=activation
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Train and validate
        best_val_loss, _, _ = train_and_validate(
            model, train_loader, val_loader, optimizer, criterion, scheduler, patience
        )
        
        return best_val_loss
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {e}")
        # Return a high loss value for failed trials
        return float('inf')



def run_neural_architecture_search(n_trials=50):
    """Run the Neural Architecture Search process."""

    study_name = "lstm_ransomware_nas"
    storage_name = f"sqlite:///{study_name}.db"
    
    # Create or load an existing study
    try:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="minimize",
            load_if_exists=True
        )
        logger.info(f"Study loaded/created: {study_name}")
    except Exception as e:
        logger.error(f"Error creating/loading study: {e}")
        # Fallback to in-memory study
        study = optuna.create_study(direction="minimize")
        logger.info("Using in-memory study instead")
    
    logger.info(f"Starting Neural Architecture Search with {n_trials} trials")
    start_time = time.time()
    
    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        logger.info("NAS interrupted by user")
    
    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60
    
    logger.info(f"NAS completed in {duration_minutes:.2f} minutes")
    
    # Print results
    logger.info("Best trial:")
    trial = study.best_trial
    
    logger.info(f"  Value (validation loss): {trial.value:.5f}")
    logger.info("  Params:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Visualize results
    try:
        # Optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_PATH, 'nas_optimization_history.png'))
        plt.close()
        
        # Parameter importances
        plt.figure(figsize=(12, 10))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_PATH, 'nas_param_importances.png'))
        plt.close()
        
        # Parallel coordinate plot for parameters
        plt.figure(figsize=(20, 10))
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_PATH, 'nas_parallel_coordinate.png'))
        plt.close()
    except Exception as e:
        logger.warning(f"Error creating visualizations: {e}")
    
    return study

# =============================================================================
# Train the Best Model Found by NAS
# =============================================================================
def train_best_model(study, epochs=150):
    """Train the best model found by Neural Architecture Search."""

    best_params = study.best_params
    
    # Extract model parameters
    num_lstm_layers = best_params["num_lstm_layers"]
    lstm_hidden_sizes = [best_params["first_hidden_size"]]
    
    for i in range(1, num_lstm_layers):
        lstm_hidden_sizes.append(best_params[f"lstm_hidden_size_{i}"])
    
    lstm_dropout = best_params["lstm_dropout"]
    bidirectional = best_params["bidirectional"]
    
    num_fc_layers = best_params["num_fc_layers"]
    fc_layers = []
    
    for i in range(num_fc_layers):
        fc_layers.append(best_params[f"fc_size_{i}"])
    
    fc_dropout = best_params["fc_dropout"]
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    weight_decay = best_params["weight_decay"]
    use_batch_norm = best_params["use_batch_norm"]
    activation = best_params["activation"]
    patience = best_params["patience"]
    
    # Log the best architecture
    logger.info("Best architecture parameters:")
    logger.info(f"LSTM layers: {num_lstm_layers}, Hidden sizes: {lstm_hidden_sizes}")
    logger.info(f"Bidirectional: {bidirectional}, Dropout: {lstm_dropout}")
    logger.info(f"FC layers: {fc_layers}, FC dropout: {fc_dropout}")
    logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    logger.info(f"Weight decay: {weight_decay}, Batch norm: {use_batch_norm}")
    logger.info(f"Activation: {activation}, Patience: {patience}")
    
    # Load data
    train_loader, val_loader, test_loader = load_data(batch_size)
    
    # Create model with best architecture
    best_model = DynamicLSTMRansomwareDetector(
        input_size=23,
        hidden_sizes=lstm_hidden_sizes,
        num_layers=num_lstm_layers,
        dropout=lstm_dropout,
        bidirectional=bidirectional,
        fc_layers=fc_layers,
        fc_dropout=fc_dropout,
        use_batch_norm=use_batch_norm,
        activation=activation
    ).to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    logger.info(f"Model parameter count: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Train model
    model_path = os.path.join(MODELS_PATH, "best_nas_lstm_model.pth")
    _, train_losses, val_losses = train_and_validate(
        best_model, train_loader, val_loader, optimizer, criterion, scheduler, 
        patience, epochs=epochs, model_path=model_path
    )
    
    # Plot training curves
    loss_curve_path = os.path.join(MODELS_PATH, 'best_model_training_curve.png')
    plot_loss_curves(train_losses, val_losses, loss_curve_path)
    
    # Load best model for evaluation
    best_model.load_state_dict(torch.load(model_path))
    
    # Evaluate on test set
    test_results = evaluate_model(best_model, test_loader)
    
    # Log evaluation results
    logger.info("\n===== Test Set Evaluation =====")
    logger.info(f"Test Loss: {test_results['loss']:.4f}")
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_results['precision']:.4f}")
    logger.info(f"Test Recall: {test_results['recall']:.4f}")
    logger.info(f"Test F1 Score: {test_results['f1']:.4f}")
    logger.info("\nConfusion Matrix:")
    logger.info(test_results['confusion_matrix'])
    logger.info("\nClassification Report:")
    logger.info(test_results['classification_report'])
    
    # Plot confusion matrix
    cm_path = os.path.join(MODELS_PATH, 'best_model_confusion_matrix.png')
    plot_confusion_matrix(test_results['confusion_matrix'], cm_path)
    
    # Save architecture details and results
    with open(os.path.join(MODELS_PATH, 'best_architecture.txt'), 'w') as f:
        f.write("Best LSTM Model Architecture:\n\n")
        f.write(f"Number of LSTM Layers: {num_lstm_layers}\n")
        f.write(f"LSTM Hidden Sizes: {lstm_hidden_sizes}\n")
        f.write(f"Bidirectional: {bidirectional}\n")
        f.write(f"LSTM Dropout: {lstm_dropout}\n")
        f.write(f"\nNumber of FC Layers: {num_fc_layers}\n")
        f.write(f"FC Layer Sizes: {fc_layers}\n")
        f.write(f"FC Dropout: {fc_dropout}\n")
        f.write(f"\nBatch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Use Batch Normalization: {use_batch_norm}\n")
        f.write(f"Activation Function: {activation}\n")
        f.write(f"Early Stopping Patience: {patience}\n")
        f.write(f"Total Parameters: {num_params:,}\n")
        
        f.write(f"\nTest Metrics:\n")
        f.write(f"Test Loss: {test_results['loss']:.4f}\n")
        f.write(f"Test Accuracy: {test_results['accuracy']:.4f}\n")
        f.write(f"Test Precision: {test_results['precision']:.4f}\n")
        f.write(f"Test Recall: {test_results['recall']:.4f}\n")
        f.write(f"Test F1 Score: {test_results['f1']:.4f}\n")
        
        f.write(f"\nConfusion Matrix:\n")
        f.write(str(test_results['confusion_matrix']))
        f.write(f"\n\nClassification Report:\n")
        f.write(test_results['classification_report'])
    
    return best_model, test_results

# =============================================================================
# Run a Simple Test Model
# =============================================================================
def run_simple_test_model():
    """Run a simple model to verify that the data and pipeline work correctly."""

    logger.info("Running simple test model to verify pipeline...")
    
    # Load data
    train_loader, val_loader, test_loader = load_data(batch_size=32)
    
    # Create a simple model
    model = DynamicLSTMRansomwareDetector(
        input_size=23,
        hidden_sizes=[64],
        num_layers=1,
        dropout=0.3,
        bidirectional=False,
        fc_layers=[32],
        fc_dropout=0.3,
        use_batch_norm=True,
        activation='relu'
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Train for a few epochs to verify
    _, train_losses, val_losses = train_and_validate(
        model, train_loader, val_loader, optimizer, criterion, scheduler, 
        patience=5, epochs=10
    )
    
    # Evaluate
    test_results = evaluate_model(model, test_loader)
    
    logger.info(f"Simple test model - Test Accuracy: {test_results['accuracy']:.4f}, Test Loss: {test_results['loss']:.4f}")
    
    return test_results


# =============================================================================
# Main Function
# =============================================================================
def main():
    # First run a simple model to verify the pipeline works
    logger.info("==== Running Simple Test Model ====")
    run_simple_test_model()
    
    # Define the number of trials for the search
    n_trials = 100  
    
    # Run NAS
    logger.info("\n==== Starting Neural Architecture Search ====")
    study = run_neural_architecture_search(n_trials)
    
    # Train and evaluate the best model
    logger.info("\n==== Training Best Model Found by NAS ====")
    best_model, test_results = train_best_model(study)
    
    logger.info("\n==== NAS Pipeline Completed Successfully ====")
    logger.info(f"Best model saved to: {os.path.join(MODELS_PATH, 'best_nas_lstm_model.pth')}")
    logger.info(f"Final test accuracy: {test_results['accuracy']:.4f}")
    
    return best_model


if __name__ == "__main__":
    main()