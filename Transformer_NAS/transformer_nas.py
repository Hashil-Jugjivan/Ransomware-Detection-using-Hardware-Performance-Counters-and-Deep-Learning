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
import random
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup

# Set paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Preprocessing import normalization_small

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set paths
DATA_PATH = r"../Transformer_NAS"
MODELS_PATH = r"../Models/nas_transformer_ransomware_model"
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)

# Set manual seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# =============================================================================
# Data Augmentation Functions
# =============================================================================
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
# Data Loading Functions
# =============================================================================
def check_and_preprocess_data():
    """Check if data exists, and if not, preprocess it."""

    if not os.path.exists(os.path.join(DATA_PATH, "train.npy")):
        logger.info("Preprocessing data...")
        normalization_small.normalize_balanced_dataset(
            dataset_path=r"../Dataset", 
            save_directory=DATA_PATH, 
            norm=False, 
            num_samples=288
        )

def load_data(batch_size=32):
    """Load normalized data from numpy files and return train, validation, and test DataLoaders."""

    # Load datasets
    X_train = np.load(os.path.join(DATA_PATH, "train.npy"))
    y_train = np.load(os.path.join(DATA_PATH, "train_labels.npy"))
    X_val = np.load(os.path.join(DATA_PATH, "val.npy"))
    y_val = np.load(os.path.join(DATA_PATH, "val_labels.npy"))
    X_test = np.load(os.path.join(DATA_PATH, "test.npy"))
    y_test = np.load(os.path.join(DATA_PATH, "test_labels.npy"))

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)  # [batch, seq, features]
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    # Squeeze labels to get shape [batch]
    y_train = torch.tensor(y_train, dtype=torch.float32).squeeze()
    y_val = torch.tensor(y_val, dtype=torch.float32).squeeze()
    y_test = torch.tensor(y_test, dtype=torch.float32).squeeze()

    # Print shapes for verification
    logger.info(f"Data shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    logger.info(f"Label counts: Train (0={torch.sum(y_train==0)}, 1={torch.sum(y_train==1)}), "
                f"Val (0={torch.sum(y_val==0)}, 1={torch.sum(y_val==1)}), "
                f"Test (0={torch.sum(y_test==0)}, 1={torch.sum(y_test==1)})")

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# =============================================================================
# Positional Encoding
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
# Dynamic Transformer Model
# =============================================================================
class DynamicTransformerRansomwareDetector(nn.Module):
    def __init__(self, 
                 num_features=23, 
                 seq_len=60, 
                 d_model=64, 
                 nhead=4, 
                 num_layers=2, 
                 dim_feedforward=128,
                 dropout=0.1,
                 activation="relu",
                 layer_norm_eps=1e-5,
                 use_positional_encoding=True,
                 fc_sizes=[64, 32],
                 use_residual=True,
                 pooling_type="mean"):
        super().__init__()
        
        # Model architecture parameters
        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        self.use_positional_encoding = use_positional_encoding
        self.use_residual = use_residual
        self.pooling_type = pooling_type
        
        # Input projection layer
        self.input_proj = nn.Linear(num_features, d_model)
        
        # Positional encoding layer
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=True
            )
            encoder_layers.append(encoder_layer)
        
        self.transformer_encoder_layers = nn.ModuleList(encoder_layers)
        
        # Pooling layer
        if pooling_type == "adaptive_avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == "adaptive_max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers
        fc_layers = []
        in_features = d_model
        
        for fc_size in fc_sizes:
            fc_layers.append(nn.Linear(in_features, fc_size))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            in_features = fc_size
        
        fc_layers.append(nn.Linear(in_features, 1))
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x, attention_mask=None):
        # x: [batch, seq_len, num_features]
        
        # Check input shapes
        batch_size, seq_len, num_features = x.shape
        assert num_features == self.num_features, f"Expected {self.num_features} features, got {num_features}"
        
        # Project input to model dimension
        x = self.input_proj(x)  # [batch, seq_len, d_model]
        
        # Apply positional encoding if specified
        if self.use_positional_encoding:
            x = self.pos_encoder(x)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (x.sum(dim=-1) != 0)  # [batch, seq_len]
        
        # Apply transformer layers with optional residual connections
        for layer in self.transformer_encoder_layers:
            if self.use_residual:
                out = layer(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
                x = x + out  # Residual connection
            else:
                x = layer(x, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        
        # Apply pooling based on specified type
        if self.pooling_type == "mean":
            # Mean pooling along sequence dimension
            x = x.mean(dim=1)  # [batch, d_model]
        elif self.pooling_type == "max":
            # Max pooling along sequence dimension
            x = x.max(dim=1)[0]  # [batch, d_model]
        elif self.pooling_type == "cls":
            # Use the first token (CLS token) 
            x = x[:, 0, :]  # [batch, d_model]
        elif self.pooling_type in ["adaptive_avg", "adaptive_max"]:
            # Adaptive pooling requires dimension rearrangement
            x = x.transpose(1, 2)  # [batch, d_model, seq_len]
            x = self.pool(x).squeeze(-1)  # [batch, d_model]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        # Apply sigmoid for binary classification
        return torch.sigmoid(x).squeeze()

# =============================================================================
# Training and Evaluation Functions
# =============================================================================
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, patience, epochs=100, model_path=None, use_augmentation=True, clip_grad_norm=1.0):
    """
    Train and validate the transformer model with early stopping.
    Optionally applies data augmentation and gradient clipping.
    """

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            # Apply data augmentation during training if specified
            if use_augmentation:
                inputs = augment_time_series(inputs)
            
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Generate attention mask
            attention_mask = (inputs.sum(dim=-1) != 0).to(device)
            
            # Forward pass, loss calculation, and backpropagation
            optimizer.zero_grad()
            outputs = model(inputs, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Apply gradient clipping if specified
            if clip_grad_norm > 0:
                clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
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
                attention_mask = (inputs.sum(dim=-1) != 0).to(device)
                outputs = model(inputs, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model 
            if model_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, model_path)
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
                        f"Val Loss: {avg_val_loss:.4f}")
    
    return best_val_loss, train_losses, val_losses

def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test set and calculate performance metrics.
    """

    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    test_loss = 0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            attention_mask = (inputs.sum(dim=-1) != 0).to(device)
            outputs = model(inputs, attention_mask)
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
    """
    Objective function for Optuna to optimize model architecture and hyperparameters.
    """
    try:
        # Sample hyperparameters
        
        # Transformer architecture parameters
        d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128, 256, 512])
        nhead_options = [2, 4, 8]  # Ensure nhead divides d_model evenly
        valid_nheads = [h for h in nhead_options if d_model % h == 0]
        nhead = trial.suggest_categorical("nhead", valid_nheads)
        num_layers = trial.suggest_int("num_layers", 1, 6)
        dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512])
        
        # Regularization parameters
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        use_positional_encoding = trial.suggest_categorical("use_positional_encoding", [True, False])
        use_residual = trial.suggest_categorical("use_residual", [True, False])
        
        # Pooling strategy
        pooling_type = trial.suggest_categorical("pooling_type", ["mean", "max", "cls", "adaptive_avg", "adaptive_max"])
        
        # FC layers configuration
        num_fc_layers = trial.suggest_int("num_fc_layers", 0, 3)
        fc_sizes = []
        if num_fc_layers > 0:
            prev_size = trial.suggest_categorical("fc_size_0", [32, 64, 128])
            fc_sizes.append(prev_size)
            
            for i in range(1, num_fc_layers):
                # Each subsequent layer has equal or fewer units
                size_options = [s for s in [16, 32, 64, 128] if s <= prev_size]
                if not size_options:  # Fallback if no valid options
                    size_options = [16]
                fc_size = trial.suggest_categorical(f"fc_size_{i}", size_options)
                fc_sizes.append(fc_size)
                prev_size = fc_size
        
        # Learning hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        activation = trial.suggest_categorical("activation", ["relu", "gelu"])
        
        # Training options
        use_augmentation = trial.suggest_categorical("use_augmentation", [True, False])
        clip_grad_norm_val = trial.suggest_categorical("clip_grad_norm", [0.0, 0.5, 1.0, 5.0])
        patience = trial.suggest_int("patience", 10, 20)
        use_scheduler = trial.suggest_categorical("use_scheduler", [True, False])
        
        # Print the sampled parameters for this trial
        logger.info(f"Trial {trial.number} - Hyperparameters:")
        logger.info(f"  Transformer architecture: d_model={d_model}, nhead={nhead}, layers={num_layers}, ff_dim={dim_feedforward}")
        logger.info(f"  Regularization: dropout={dropout}, pos_encoding={use_positional_encoding}, residual={use_residual}")
        logger.info(f"  Pooling: {pooling_type}, FC layers: {fc_sizes}")
        logger.info(f"  Batch size: {batch_size}, LR: {learning_rate}, Weight decay: {weight_decay}")
        logger.info(f"  Activation: {activation}, Augmentation: {use_augmentation}, Grad clip: {clip_grad_norm_val}")
        
        # Load data
        train_loader, val_loader, _ = load_data(batch_size)
        
        # Create model
        model = DynamicTransformerRansomwareDetector(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            use_positional_encoding=use_positional_encoding,
            fc_sizes=fc_sizes,
            use_residual=use_residual,
            pooling_type=pooling_type
        ).to(device)
        
        # Set up optimizer and loss function
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Set up scheduler if specified
        scheduler = None
        if use_scheduler:
            num_training_steps = 100 * len(train_loader)  # Maximum possible steps
            num_warmup_steps = num_training_steps // 10  # 10% of total steps for warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        
        # Train and validate
        best_val_loss, _, _ = train_and_validate(
            model, 
            train_loader, 
            val_loader, 
            optimizer, 
            criterion, 
            scheduler, 
            patience,
            use_augmentation=use_augmentation,
            clip_grad_norm=clip_grad_norm_val
        )
        
        return best_val_loss
    
    except Exception as e:
        logger.error(f"Error in trial {trial.number}: {e}")
        return float('inf')

def run_neural_architecture_search(n_trials=50):
    """Run the Neural Architecture Search process."""

    study_name = "transformer_ransomware_nas"
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
    d_model = best_params["d_model"]
    nhead = best_params["nhead"]
    num_layers = best_params["num_layers"]
    dim_feedforward = best_params["dim_feedforward"]
    dropout = best_params["dropout"]
    use_positional_encoding = best_params["use_positional_encoding"]
    use_residual = best_params["use_residual"]
    pooling_type = best_params["pooling_type"]
    activation = best_params["activation"]
    
    # Extract FC layer sizes
    num_fc_layers = best_params["num_fc_layers"]
    fc_sizes = []
    for i in range(num_fc_layers):
        fc_sizes.append(best_params[f"fc_size_{i}"])
    
    # Extract training parameters
    batch_size = best_params["batch_size"]
    learning_rate = best_params["learning_rate"]
    weight_decay = best_params["weight_decay"]
    use_augmentation = best_params["use_augmentation"]
    clip_grad_norm_val = best_params["clip_grad_norm"]
    patience = best_params["patience"]
    use_scheduler = best_params["use_scheduler"]
    
    # Log the best architecture
    logger.info("Best architecture parameters:")
    logger.info(f"Transformer layers: {num_layers}, d_model: {d_model}, nhead: {nhead}")
    logger.info(f"Feedforward dim: {dim_feedforward}, Activation: {activation}")
    logger.info(f"Dropout: {dropout}, Positional encoding: {use_positional_encoding}")
    logger.info(f"Residual connections: {use_residual}, Pooling: {pooling_type}")
    logger.info(f"FC layers: {fc_sizes}")
    logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    logger.info(f"Weight decay: {weight_decay}, Augmentation: {use_augmentation}")
    logger.info(f"Gradient clipping: {clip_grad_norm_val}, Early stopping patience: {patience}")
    logger.info(f"Use scheduler: {use_scheduler}")
    
    # Load data
    train_loader, val_loader, test_loader = load_data(batch_size)
    
    # Create model with best architecture
    best_model = DynamicTransformerRansomwareDetector(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        use_positional_encoding=use_positional_encoding,
        fc_sizes=fc_sizes,
        use_residual=use_residual,
        pooling_type=pooling_type
    ).to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in best_model.parameters() if p.requires_grad)
    logger.info(f"Model parameter count: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(best_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Set up scheduler if specified
    scheduler = None
    if use_scheduler:
        num_training_steps = epochs * len(train_loader)
        num_warmup_steps = num_training_steps // 10  # 10% of total steps for warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    # Train model
    model_path = os.path.join(MODELS_PATH, "best_nas_transformer_model.pth")
    _, train_losses, val_losses = train_and_validate(
        best_model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        scheduler, 
        patience, 
        epochs=epochs, 
        model_path=model_path,
        use_augmentation=use_augmentation,
        clip_grad_norm=clip_grad_norm_val
    )
    
    # Plot training curves
    loss_curve_path = os.path.join(MODELS_PATH, 'best_model_training_curve.png')
    plot_loss_curves(train_losses, val_losses, loss_curve_path)
    
    # Load best model for evaluation
    checkpoint = torch.load(model_path)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    
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
        f.write("Best Transformer Model Architecture:\n\n")
        f.write(f"Number of Transformer Layers: {num_layers}\n")
        f.write(f"Model Dimension (d_model): {d_model}\n")
        f.write(f"Number of Attention Heads (nhead): {nhead}\n")
        f.write(f"Feedforward Dimension: {dim_feedforward}\n")
        f.write(f"Activation Function: {activation}\n")
        f.write(f"Dropout Rate: {dropout}\n")
        f.write(f"Use Positional Encoding: {use_positional_encoding}\n")
        f.write(f"Use Residual Connections: {use_residual}\n")
        f.write(f"Pooling Strategy: {pooling_type}\n")
        f.write(f"\nFC Layer Sizes: {fc_sizes}\n")
        f.write(f"\nTraining Parameters:\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Weight Decay: {weight_decay}\n")
        f.write(f"Use Data Augmentation: {use_augmentation}\n")
        f.write(f"Gradient Clipping Value: {clip_grad_norm_val}\n")
        f.write(f"Early Stopping Patience: {patience}\n")
        f.write(f"Use Learning Rate Scheduler: {use_scheduler}\n")
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
    model = DynamicTransformerRansomwareDetector(
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.1,
        activation="relu",
        use_positional_encoding=True,
        fc_sizes=[32],
        use_residual=True,
        pooling_type="mean"
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Train for a few epochs to verify
    _, train_losses, val_losses = train_and_validate(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        None, 
        patience=5, 
        epochs=10,
        use_augmentation=True,
        clip_grad_norm=1.0
    )
    
    # Evaluate
    test_results = evaluate_model(model, test_loader)
    
    logger.info(f"Simple test model - Test Accuracy: {test_results['accuracy']:.4f}, Test Loss: {test_results['loss']:.4f}")
    
    return test_results

def main():
    # Check if preprocessed data exists, and if not, preprocess it
    check_and_preprocess_data()
    
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
    logger.info(f"Best model saved to: {os.path.join(MODELS_PATH, 'best_nas_transformer_model.pth')}")
    logger.info(f"Final test accuracy: {test_results['accuracy']:.4f}")
    
    return best_model

if __name__ == "__main__":
    main()
