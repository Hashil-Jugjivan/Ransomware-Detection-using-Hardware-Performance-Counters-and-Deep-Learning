import os
import sys
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import argparse
import numpy as np 
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
# Set multiprocessing sharing strategy
torch.multiprocessing.set_sharing_strategy('file_system')

# Import the normalization file
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Data_Preprocessing import norm_small_noval

torch.manual_seed(7) # enable repeating the results, original seed was 7
# device = torch.device("cuda:0" if use_cuda else "cpu")

###############################
# --- Model Architecture --- #
###############################

class ResCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout):
        super(ResCNN, self).__init__()

        self.op = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels, out_channels, kernel, stride, padding=kernel//2),
            # nn.BatchNorm1d(out_channels),
            # nn.GELU(),
            # nn.Dropout(dropout),
            # nn.Conv1d(out_channels, out_channels, kernel, stride, padding=kernel//2),
        )

    def forward(self, x):
        # residual = x
        x = self.op(x)  ## number of feature equals to the channel size, which change from 3->32->64
        # x += residual
        return x  # (batch, channel, sequence_length)


class BiGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BiGRU, self).__init__()

        self.bigru = nn.GRU(input_size=rnn_dim, hidden_size=hidden_size, \
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.bigru(x)
        x = self.dropout(x)
        return x

class ToolModule(nn.Module):
    def __init__(self, cnn_layers, rnn_layers, feat_dim, rnn_dim, n_class, dropout, input_len, input_height): # feat_dim = 64, rnn_dim =128
        super(ToolModule, self).__init__()

        # Stem CNN expects input with 1 channel 
        self.stemcnn = nn.Conv2d(1, feat_dim, 3, stride=1, padding=1)
        # self.rescnn = nn.Sequential(
        #     *[ResCNN(feat_dim, feat_dim, 7, stride=1, dropout=dropout) for _ in range(cnn_layers)]
        # )
        self.fc = nn.Linear(feat_dim * input_height, rnn_dim)
        self.rnn = nn.Sequential(
            *[BiGRU(rnn_dim = rnn_dim if i == 0 else rnn_dim*2, \
                hidden_size = rnn_dim, dropout = dropout, batch_first=True) \
                    for i in range(rnn_layers)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            # nn.Linear(rnn_dim, n_class)
        )
        self.fc2 = nn.Linear(rnn_dim*input_len, n_class)

    def forward(self, x): 
        x = self.stemcnn(x)
        sizes = x.size()
        # print(x.size())
        x = x.view(sizes[0], sizes[1]*sizes[2], sizes[3])
        # x = self.rescnn(x) # x = [batch, channel (features), length]
        x = x.transpose(1, 2) # x = [batch, length, channel (features)]
        x = self.fc(x)
        # print(x.size())
        x = self.rnn(x)
        # print(x.size())
        x = self.classifier(x)
        # print(x.size())
        sizes = x.size()
        x = x.view(sizes[0], sizes[1]*sizes[2])
        x = self.fc2(x)
        return x

class_mapping = {1: 1, 0: 0}
def map_classes(y, mapping):
    return [mapping[c.item()] for c in y]

def train(model, train_dataloader, criterion, optimizer, scheduler, epoch, device):
    print("\nstart training!")
    model.train()

    # intialize accumulators
    num_batch = 0
    total_loss = 0
    total_acc = 0
    total_c = 0
    correct_c = 0

    for input_sequences, labels in tqdm(train_dataloader, ncols=100):
        num_batch += 1

        # input_sequences shape: [batch, 60, 23]; add channel dim → [batch, 1, 60, 23]
        input_sequences = input_sequences.unsqueeze(1).to(device)
        labels = labels.to(device).long() # ensure labels are LongTensors

        optimizer.zero_grad()
        output = model(input_sequences) # output shape : [batch, n_class]
        output = F.log_softmax(output, dim=1)
        pred = torch.argmax(output,dim=1)
        loss = criterion(output, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_acc = (pred == labels).sum().item() / labels.size(0)
        total_acc += batch_acc

        # For combined accuracy, using map_classes 
        mapped_labels = map_classes(labels, class_mapping)
        mapped_predicted = map_classes(pred, class_mapping)
        total_c += labels.size(0)
        correct_c += sum(mapped_predicted[i] == mapped_labels[i] for i in range(len(mapped_labels)))


    avg_loss = total_loss / num_batch
    avg_acc = total_acc / num_batch
    combined_acc = correct_c / total_c
    print("Train Epoch: {}, Average Loss: {:.6f}, Acc: {:.6f}, Combined Acc: {:.6f}".format(epoch, avg_loss, avg_acc, combined_acc))
    return avg_loss, avg_acc, combined_acc


def test(model, test_dataloader, criterion, epoch, device, args):
    print("\nStart evaluation!")
    model.eval()
    total_loss = 0
    num_batch = 0
    total_acc = 0
    total_c_acc, correct_c_acc = 0, 0

    # Initialize per-class accuracy counts for binary classification
    num_classes = args.n_class
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    
    with torch.no_grad():
        for input_sequences, labels in tqdm(test_dataloader, ncols=100):
            num_batch += 1
            input_sequences = input_sequences.unsqueeze(1).to(device)
            labels = labels.to(device).long()

            output = model(input_sequences)
            output = F.log_softmax(output, dim=1)
            pred = torch.argmax(output, dim=1)
            loss = criterion(output, labels)
            total_loss += loss.item()

            batch_acc = (pred == labels).sum().item() / labels.size(0)
            total_acc += batch_acc

            # Per-class accuracy calculation
            for i in range(num_classes):
                class_idx = (labels == i)
                correct = (pred[class_idx] == i).sum().item()
                total = class_idx.sum().item()
                class_correct[i] += correct
                class_total[i] += total

            # Combined accuracy (same as standard in binary classification)
            mapped_labels = map_classes(labels, class_mapping)
            mapped_predicted = map_classes(pred, class_mapping)
            total_c_acc += labels.size(0)
            correct_c_acc += sum(mapped_predicted[i] == mapped_labels[i] for i in range(len(mapped_labels)))
    
    avg_loss = total_loss / num_batch
    avg_acc = total_acc / num_batch
    combined_acc = correct_c_acc / total_c_acc
    print("Test Epoch: {}, Loss: {:.6f}, Acc: {:.6f}".format(epoch, avg_loss, avg_acc))
    for i in range(num_classes):
        class_acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print("Class {} Accuracy: {:.6f}".format(i, class_acc))
    print("Total Combined Accuracy: {:.6f}".format(combined_acc))
    return avg_loss, avg_acc, combined_acc

##########################################
# --- New: Confusion Matrix and Metrics ---#
##########################################
def compute_and_plot_confusion_matrix(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_sequences, labels in dataloader:
            input_sequences = input_sequences.unsqueeze(1).to(device)
            labels = labels.to(device).long()
            output = model(input_sequences)
            output = F.log_softmax(output, dim=1)
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    cr = classification_report(all_labels, all_preds, target_names=class_names)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)
    print("Accuracy: {:.4f}".format(acc))
    print("Precision: {:.4f}".format(prec))
    print("Recall: {:.4f}".format(rec))
    print("F1 Score: {:.4f}".format(f1))
    
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


def gpu_setup(use_gpu, gpu_id):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
        # torch.cuda.set_device(int(gpu_id))
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device

##########################################
# --- Main Function: Data Integration ---#
##########################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_class', type=int, default=2, help='number of final classes.')
    parser.add_argument('--cnn_layers', type=int, default=3, help='number of cnn layers.')
    parser.add_argument('--rnn_layers', type=int, default=5, help='number of rnn layers.')
    parser.add_argument('--rnn_dim', type=int, default=128, help='dimension of rnn input.')
    parser.add_argument('--n_feats', type=int, default=32, help='number of input features.')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)  
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--gpu_id', type=str, default="0")
    # load_path can be used to load a saved model checkpoint if needed
    parser.add_argument('--load_path', type=str, default='')  
    args = parser.parse_args()

    # --- GPU Setup ---
    device = gpu_setup(True, args.gpu_id)

    # --- Data Directories ---
    dataset_path = r"../Dataset"    
    data_directory = r"../CNN_RNN"    
    model_save_directory = r"../Models/cnn_rnn_ransomware_model"  
    
    if not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)

    from Data_Preprocessing.norm_small_noval import normalize_balanced_dataset

    # --- Normalize and Save the Data ---
    # This function loads the CSV files, normalizes them, and saves them as numpy files
    norm_small_noval.normalize_balanced_dataset(dataset_path=dataset_path, save_directory=data_directory, norm=False, test_ratio=0.2, num_samples=288)

    # --- Load Preprocessed Data ---
    X_train = np.load(os.path.join(data_directory, "train.npy"))
    y_train = np.load(os.path.join(data_directory, "train_labels.npy"))
    X_test = np.load(os.path.join(data_directory, "test.npy"))
    y_test = np.load(os.path.join(data_directory, "test_labels.npy"))

    # Convert numpy arrays to torch tensors.
    # Our normalized data have shape [samples, 60, 23] 
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).squeeze()  # Ensure shape [samples]
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).squeeze()

    # Create DataLoaders
    train_dataloader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Determine input length ---
    # For our dataset: each sample is 60 rows. 
    # Here we simply use the time dimension (60) as our input length.
    input_height = 60 # number of rows
    input_len = 23 # number of columns (sequence length)
    # --- Instantiate the Model ---
    model = ToolModule(args.cnn_layers, args.rnn_layers, args.n_feats, args.rnn_dim, args.n_class, args.dropout, input_len, input_height).to(device)
    print("Number of model parameters:", sum([param.nelement() for param in model.parameters()]))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # --- Load Checkpoint if available ---
    start_epoch = 0
    if args.load_path and os.path.exists(os.path.join(model_save_directory, args.load_path)):
        checkpoint = torch.load(os.path.join(model_save_directory, args.load_path))
        start_epoch = checkpoint['start_epoch']
        model.load_state_dict(checkpoint['base-model'])
        print("Checkpoint loaded from", args.load_path)

    # --- Training Loop ---
    best_acc = 0
    best_acc_c = 0
    train_acc_list = []
    train_loss_list = []
    best_test_acc = 0
    test_acc_list = []
    test_loss_list = []
    test_combined_acc_list = []

    # Modify the save_path prefix to reflect our HPC model
    save_path = "detect_hpc_" + "rnn_layer_" + str(args.rnn_layers) + "_rnndim" + str(args.rnn_dim) + "n_feats" + str(args.n_feats) + ".pt"

    for epoch in range(start_epoch, args.epochs):
        print("Epoch: {}/{}".format(epoch, args.epochs))
        start_time = time.time()
        train_loss, train_acc, train_acc_c = train(model, train_dataloader, criterion, optimizer, scheduler, epoch, device)
        end_time = time.time()
        print("Training for epoch {} took {} seconds".format(epoch, end_time - start_time))
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # Save checkpoint if training accuracy improves (after a few epochs)
        save_checkpoint = False
        if (train_acc > best_acc) and epoch > 4:
            best_acc = train_acc
            save_name = 'acc_' + str(best_acc) + '_' + save_path
            torch.save({
                'start_epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'base-model': model.state_dict(),
                'best_acc': best_acc,
                'train_acc_list': train_acc_list,
                'train_loss_list': train_loss_list
            }, os.path.join(model_save_directory, save_name))
            save_checkpoint = True

        # Evaluate on test set if training accuracy is high enough
        if train_acc > 0.85:
            print("--- Evaluating on test set ---")
            start_time = time.time()
            test_loss, test_acc, test_acc_c = test(model, test_dataloader, criterion, epoch, device, args)
            end_time = time.time()
            print("Testing for epoch {} took {} seconds".format(epoch, end_time - start_time))
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            test_combined_acc_list.append(test_acc_c)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                save_name = 'test_acc_' + str(best_test_acc) + '_' + save_path
                torch.save({
                    'start_epoch': epoch + 1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'base-model': model.state_dict(),
                    'best_acc': best_acc,
                    'train_acc_list': train_acc_list,
                    'train_loss_list': train_loss_list
                }, os.path.join(model_save_directory, save_name))
                print('Best test accuracy updated: {:.6f}'.format(best_test_acc))
            if test_acc_c == max(test_combined_acc_list):
                if not save_checkpoint:
                    save_name = 'test_acc_c_' + str(test_acc_c) + '_' + save_path
                    torch.save({
                        'start_epoch': epoch + 1,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'base-model': model.state_dict(),
                        'best_acc': best_acc,
                        'train_acc_list': train_acc_list,
                        'train_loss_list': train_loss_list
                    }, os.path.join(model_save_directory, save_name))
                    print("Checkpoint saved to", save_name)
        
        # Optional early stopping condition (if training accuracy is extremely high)
        if best_acc >= 0.995:
            break

    print("--- Final Evaluation with Best Training Accuracy Model ---")
    best_path = 'acc_' + str(best_acc) + '_' + save_path
    if os.path.exists(os.path.join(model_save_directory, best_path)):
        checkpoint = torch.load(os.path.join(model_save_directory, best_path))
        model.load_state_dict(checkpoint['base-model'])
    start_time = time.time()
    test_loss, test_acc, test_acc_c = test(model, test_dataloader, criterion, epoch, device, args)
    end_time = time.time()
    print("Final testing took {} seconds".format(end_time - start_time))
    train_loss_list.append(test_loss)
    train_acc_list.append(test_acc)
    print('Best Training Loss: {:.6f}'.format(min(train_loss_list)))
    print('Best Training Accuracy: {:.6f}'.format(best_acc))
    print('Best Test Loss: {:.6f}'.format(min(test_loss_list)))
    print('Best Test Accuracy: {:.6f}'.format(best_test_acc))
    print('Best Combined Test Accuracy: {:.6f}'.format(max(test_combined_acc_list)))


    # ----------------------------------------------------------------------
    # 1) Smoothing function to reduce large spikes in the plotted curve
    # ----------------------------------------------------------------------
    def smooth_curve(points, factor=0.8):
        smoothed_points = []
        for p in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + p * (1 - factor))
            else:
                smoothed_points.append(p)
        return smoothed_points

    # ----------------------------------------------------------------------
    # 2) Plot Train & Test Loss Curves with smoothing
    # ----------------------------------------------------------------------
    if len(train_loss_list) > 0 and len(test_loss_list) > 0:
        train_loss_smooth = smooth_curve(train_loss_list, factor=0.8)
        test_loss_smooth = smooth_curve(test_loss_list, factor=0.8)

        plt.figure()
        plt.plot(range(len(train_loss_smooth)), train_loss_smooth, label="Train Loss (Smoothed)")
        plt.plot(range(len(test_loss_smooth)), test_loss_smooth, label="Test Loss (Smoothed)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train & Test Loss Curves (Smoothed)")
        plt.legend()
        plt.savefig(os.path.join(model_save_directory, "loss_curves.png"))
        plt.show()

    # ----------------------------------------------------------------------
    # 3) Compute and Plot Confusion Matrix
    # ----------------------------------------------------------------------
    compute_and_plot_confusion_matrix(model, test_dataloader, device, ["Safe", "Malware"])

if __name__ == "__main__":
    main()