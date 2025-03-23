import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import time
import optuna
from optuna.trial import TrialState
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import sys
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Preprocessing import normalization_small

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_PATH = r"../CNN_NAS"
MODELS_PATH = r"../Models/nas_cnn_ransomware_model"
os.makedirs(MODELS_PATH, exist_ok=True)

# Set manual seed for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Check if model data exists, if not preprocess it
if not os.path.exists(os.path.join(DATA_PATH, "train.npy")):
    logger.info("Preprocessing data...")
    normalization_small.normalize_balanced_dataset(
        dataset_path=r"../Dataset", 
        save_directory=DATA_PATH, 
        norm=False, 
        num_samples=288
    )

# Load Preprocessed Data
def load_data(batch_size=32):
    X_train = np.load(os.path.join(DATA_PATH, "train.npy"))
    y_train = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    X_val = np.load(os.path.join(DATA_PATH, "val.npy"))
    y_val = np.load(os.path.join(DATA_PATH, "val_labels.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "test_labels.npy"))

    # Reshape for Conv1D: [batch, features, sequence]
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Print shapes for verification
    logger.info(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Create DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

# Dynamic CNN Model with configurable architecture and sequence length checks
class DynamicCNNRansomwareDetector(nn.Module):
    def __init__(self, 
                 num_conv_layers=2, 
                 filters=[32, 64], 
                 kernel_sizes=[3, 3],
                 pool_sizes=[2, 2],
                 dropout_rates=[0.3, 0.3],
                 fc_layers=[256, 128],
                 fc_dropout_rates=[0.4, 0.3],
                 use_batch_norm=True):
        super(DynamicCNNRansomwareDetector, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        
        in_channels = 23  # Initial input channels (HPC features)
        sequence_length = 60  # Initial sequence length (time steps)
        
        # Validate architecture to prevent pooling issues
        # Check that sequence length will not become too small
        valid_architecture = True
        for i in range(num_conv_layers):
            # Calculate sequence length after pooling
            sequence_length = sequence_length // pool_sizes[i]
            if sequence_length <= 0:
                valid_architecture = False
                break
        
        if not valid_architecture:
            raise ValueError("Invalid architecture: sequence length becomes zero after pooling")
        
        # Create convolutional layers dynamically
        for i in range(num_conv_layers):
            out_channels = filters[i]
            kernel_size = kernel_sizes[i]
            
            self.conv_layers.append(nn.Conv1d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                padding=kernel_size//2
            ))
            
            if use_batch_norm:
                self.batch_norm_layers.append(nn.BatchNorm1d(out_channels))
                
            self.pool_layers.append(nn.MaxPool1d(kernel_size=pool_sizes[i]))
            self.dropout_layers.append(nn.Dropout(dropout_rates[i]))
            
            in_channels = out_channels
        
        # Determine size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 23, 60)
            conv_output = self.forward_conv(dummy_input)
            fc_input_size = conv_output.shape[1]
            logger.info(f"FC input size after convolutions: {fc_input_size}")
        
        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_batch_norm_layers = nn.ModuleList()
        self.fc_dropout_layers = nn.ModuleList()
        
        prev_size = fc_input_size
        for i, fc_size in enumerate(fc_layers):
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            if use_batch_norm:
                self.fc_batch_norm_layers.append(nn.BatchNorm1d(fc_size))
            self.fc_dropout_layers.append(nn.Dropout(fc_dropout_rates[i]))
            prev_size = fc_size
        
        # Output layer
        self.fc_output = nn.Linear(prev_size, 1)
        
    def forward_conv(self, x):
        # For debugging
        original_shape = x.shape
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.use_batch_norm:
                x = self.batch_norm_layers[i](x)
            x = F.relu(x)
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)
        
        x = torch.flatten(x, start_dim=1)
        return x
        
    def forward(self, x):
        x = self.forward_conv(x)
        
        # Fully connected layers
        for i, fc in enumerate(self.fc_layers):
            x = fc(x)
            if self.use_batch_norm:
                x = self.fc_batch_norm_layers[i](x)
            x = F.relu(x)
            x = self.fc_dropout_layers[i](x)
        
        x = torch.sigmoid(self.fc_output(x))
        return x.squeeze()

# Function to check if architecture is valid before creating model
def is_valid_architecture(num_conv_layers, pool_sizes):
    sequence_length = 60
    for i in range(num_conv_layers):
        sequence_length = sequence_length // pool_sizes[i]
        if sequence_length <= 0:
            return False
    return True

# Training function
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, 
                       patience, epochs=100, model_path=None):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
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
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            if model_path:
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}, "
                        f"LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}")
    
    return best_val_loss, train_losses, val_losses

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    test_loss = 0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())
            test_loss += loss.item()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((outputs.cpu().numpy() > 0.5).astype(int))
    
    accuracy = accuracy_score(y_true, y_pred)
    avg_test_loss = test_loss / len(test_loader)
    
    return accuracy, avg_test_loss, y_true, y_pred

# Objective function for Optuna
def objective(trial):
    # Sample hyperparameters for the search space
    
    # Architecture parameters
    num_conv_layers = trial.suggest_int("num_conv_layers", 1, 3)
    
    # Dynamically create lists based on number of conv layers
    filters = []
    kernel_sizes = []
    pool_sizes = []
    dropout_rates = []
    
    # Sample parameters in a way that prevents invalid architectures
    for i in range(num_conv_layers):
        filters.append(trial.suggest_int(f"filters_{i}", 16, 128, step=16))
        kernel_sizes.append(trial.suggest_int(f"kernel_size_{i}", 3, 7, step=2))
        
        # Restrict pool sizes to prevent sequence length from becoming too small
        if i == 0:
            
            pool_sizes.append(trial.suggest_int(f"pool_size_{i}", 2, 3))
        else:
            
            pool_sizes.append(2)  
            
        dropout_rates.append(trial.suggest_float(f"conv_dropout_{i}", 0.1, 0.5))
    
    # Verify the architecture is valid
    if not is_valid_architecture(num_conv_layers, pool_sizes):
        # Return a high loss value for invalid architectures
        return float('inf')
    
    # FC layers parameters
    num_fc_layers = trial.suggest_int("num_fc_layers", 1, 3)
    
    fc_layers = []
    fc_dropout_rates = []
    
    for i in range(num_fc_layers):
        fc_layers.append(trial.suggest_int(f"fc_size_{i}", 64, 512, step=64))
        fc_dropout_rates.append(trial.suggest_float(f"fc_dropout_{i}", 0.1, 0.5))
    
    # Training parameters
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    patience = trial.suggest_int("patience", 10, 20)
    use_batch_norm = trial.suggest_categorical("use_batch_norm", [True, False])
    
    try:
        # Load data with the chosen batch size
        train_loader, val_loader, _ = load_data(batch_size)
        
        # Create model with the chosen architecture
        model = DynamicCNNRansomwareDetector(
            num_conv_layers=num_conv_layers,
            filters=filters,
            kernel_sizes=kernel_sizes,
            pool_sizes=pool_sizes,
            dropout_rates=dropout_rates,
            fc_layers=fc_layers,
            fc_dropout_rates=fc_dropout_rates,
            use_batch_norm=use_batch_norm
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Train and validate model
        best_val_loss, _, _ = train_and_validate(
            model, train_loader, val_loader, optimizer, criterion, scheduler, patience
        )
        
        return best_val_loss
        
    except Exception as e:
        logger.error(f"Error in trial: {e}")
        # Return a high loss value for failed trials
        return float('inf')

# Main NAS procedure
def run_neural_architecture_search(n_trials=50):
    study_name = "cnn_ransomware_nas"
    storage_name = f"sqlite:///{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True
    )
    
    logger.info(f"Starting Neural Architecture Search with {n_trials} trials")
    start_time = time.time()
    
    study.optimize(objective, n_trials=n_trials)
    
    end_time = time.time()
    
    logger.info(f"NAS completed in {(end_time - start_time)/60:.2f} minutes")
    
    
    logger.info("Best trial:")
    trial = study.best_trial
    
    logger.info(f"  Value (validation loss): {trial.value:.5f}")
    logger.info("  Params:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Visualize optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig(os.path.join(MODELS_PATH, 'nas_optimization_history.png'))
    
    # Visualize parameter importances
    plt.figure(figsize=(10, 10))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig(os.path.join(MODELS_PATH, 'nas_param_importances.png'))
    
    return study

# Train the best model found by NAS
def train_best_model(study):
    best_params = study.best_params
    
    # Extract architecture parameters
    num_conv_layers = best_params["num_conv_layers"]
    num_fc_layers = best_params["num_fc_layers"]
    
    filters = [best_params[f"filters_{i}"] for i in range(num_conv_layers)]
    kernel_sizes = [best_params[f"kernel_size_{i}"] for i in range(num_conv_layers)]
    
    pool_sizes = [best_params[f"pool_size_{i}"] if f"pool_size_{i}" in best_params else 2 for i in range(num_conv_layers)]
    dropout_rates = [best_params[f"conv_dropout_{i}"] for i in range(num_conv_layers)]
    
    fc_layers = [best_params[f"fc_size_{i}"] for i in range(num_fc_layers)]
    fc_dropout_rates = [best_params[f"fc_dropout_{i}"] for i in range(num_fc_layers)]
    
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    weight_decay = best_params["weight_decay"]
    patience = best_params["patience"]
    use_batch_norm = best_params["use_batch_norm"]
    
    # Log the selected architecture
    logger.info("Best architecture parameters:")
    logger.info(f"Conv layers: {num_conv_layers}")
    logger.info(f"Filters: {filters}")
    logger.info(f"Kernel sizes: {kernel_sizes}")
    logger.info(f"Pool sizes: {pool_sizes}")
    
    # Load data
    train_loader, val_loader, test_loader = load_data(batch_size)
    
    # Create model with best architecture
    best_model = DynamicCNNRansomwareDetector(
        num_conv_layers=num_conv_layers,
        filters=filters,
        kernel_sizes=kernel_sizes,
        pool_sizes=pool_sizes,
        dropout_rates=dropout_rates,
        fc_layers=fc_layers,
        fc_dropout_rates=fc_dropout_rates,
        use_batch_norm=use_batch_norm
    ).to(device)
    
    # Print model summary
    logger.info(f"Model parameter count: {sum(p.numel() for p in best_model.parameters() if p.requires_grad)}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(best_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Train model
    model_path = os.path.join(MODELS_PATH, "best_nas_model.pth")
    _, train_losses, val_losses = train_and_validate(
        best_model, train_loader, val_loader, optimizer, criterion, scheduler, 
        patience, epochs=150, model_path=model_path
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODELS_PATH, 'best_model_training_curve.png'))
    
    # Load best model for evaluation
    best_model.load_state_dict(torch.load(model_path))
    
    # Evaluate on test set
    accuracy, test_loss, y_true, y_pred = evaluate_model(best_model, test_loader)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Print classification report
    report = classification_report(y_true, y_pred)
    logger.info(f"Classification Report:\n{report}")
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Safe', 'Malware'])
    plt.yticks([0, 1], ['Safe', 'Malware'])
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), 
                     horizontalalignment='center', 
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
    
    plt.savefig(os.path.join(MODELS_PATH, 'best_model_confusion_matrix.png'))
    
    # Save architecture details
    with open(os.path.join(MODELS_PATH, 'best_architecture.txt'), 'w') as f:
        f.write("Best Model Architecture:\n\n")
        f.write(f"Number of Conv Layers: {num_conv_layers}\n")
        for i in range(num_conv_layers):
            f.write(f"Conv Layer {i+1}: Filters={filters[i]}, Kernel={kernel_sizes[i]}, "
                    f"Pool={pool_sizes[i]}, Dropout={dropout_rates[i]}\n")
        
        f.write(f"\nNumber of FC Layers: {num_fc_layers}\n")
        for i in range(num_fc_layers):
            f.write(f"FC Layer {i+1}: Units={fc_layers[i]}, Dropout={fc_dropout_rates[i]}\n")
        
        f.write(f"\nBatch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Early Stopping Patience: {patience}\n")
        f.write(f"Batch Normalization: {use_batch_norm}\n")
        
        f.write(f"\nTest Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"\nClassification Report:\n{report}\n")

# Define and run a simpler model for testing purposes
def run_simple_test_model():
    """Run a simple model to verify that the data and pipeline work correctly"""
    logger.info("Running simple test model to verify pipeline...")
    
    # Load data
    train_loader, val_loader, test_loader = load_data(batch_size=32)
    
    # Create a simple model
    model = DynamicCNNRansomwareDetector(
        num_conv_layers=1,
        filters=[32],
        kernel_sizes=[3],
        pool_sizes=[2],
        dropout_rates=[0.3],
        fc_layers=[128],
        fc_dropout_rates=[0.3]
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
    accuracy, test_loss, _, _ = evaluate_model(model, test_loader)
    logger.info(f"Simple test model - Test Accuracy: {accuracy:.4f}, Test Loss: {test_loss:.4f}")

if __name__ == "__main__":
    # First run a simple test to verify the pipeline works
    run_simple_test_model()
    
    # Define the number of trials for the search
    n_trials = 50  
    
    # Run NAS
    study = run_neural_architecture_search(n_trials)
    
    # Train and evaluate the best model
    train_best_model(study)
    