import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random  # Added for random selection of files

# Function to load a random selection of files and create data and labels
def load_dataset(dataset_path, sub_path, num_samples=288):
    # Initialize empty arrays for data and labels
    x = np.array([]).reshape(0, 60, 23)  
    y = np.array([]).reshape(0, 1)
    
    # Get all available files in the given sub-path
    files = os.listdir(os.path.join(dataset_path, sub_path))
    selected_files = random.sample(files, min(num_samples, len(files)))
    
    for file in selected_files:
        # Read the CSV file
        df = pd.read_csv(os.path.join(dataset_path, sub_path, file)).to_numpy()
        
        # Reshape and append
        x = np.append(x, df.reshape(1, 60, 23), axis=0)
        
        # Assign labels: 1 for malware, 0 for safe
        if sub_path == "malware":
            y = np.append(y, [[1]], axis=0)
        else:
            y = np.append(y, [[0]], axis=0)
    
    return x, y

# Function to normalize data using Min-Max or Z-Score normalization
def data_normalization(data, standard=True):
    if standard:
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    # Flatten data for normalization, then reshape back
    original_shape = data.shape
    data = scaler.fit_transform(data.reshape(original_shape[0] * 60, 23)).reshape(original_shape)  
    return data

# Function to load, normalize, and split data into train/test/validation sets
def load_data(dataset_path, ratio, norm=False, num_samples=288):
    
    # Load only num_smaples random safe and num_smaples random malware datasets
    x_safe, y_safe = load_dataset(dataset_path, "safe", num_samples)
    x_malware, y_malware = load_dataset(dataset_path, "malware", num_samples)

    # Normalize data for both classes
    x_safe = data_normalization(x_safe, standard=norm)
    x_malware = data_normalization(x_malware, standard=norm)

    # Combine data and labels
    x = np.concatenate((x_safe, x_malware), axis=0)
    y = np.concatenate((y_safe, y_malware), axis=0)

    # Shuffle the combined data
    x, y = shuffle(x, y)

    # Split into train, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=ratio[1], random_state=8)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=ratio[2]/(1. - ratio[1]), random_state=9)

    return X_train, X_test, X_val, y_train, y_test, y_val

def display_normalized_data(X_train, X_test, X_val):
    print("\n Sample of Normalized Training Data:")
    print(pd.DataFrame(X_train[50, :, :]))  # Displaying the first sample in the training set

    print("\n Sample of Normalized Test Data:")
    print(pd.DataFrame(X_test[5, :, :]))  # Displaying the first sample in the test set

    print("\n Sample of Normalized Validation Data:")
    print(pd.DataFrame(X_val[3, :, :]))  # Displaying the first sample in the validation set)

def normalize_balanced_dataset(dataset_path=r"../Dataset", save_directory=r"../LLM", norm=False, num_samples=288):
    # Load and normalize the dataset
    ratios = [0.7, 0.2, 0.1]  # Ratios for train/test/validation
    X_train, X_test, X_val, y_train, y_test, y_val = load_data(dataset_path, ratios, norm, num_samples)

    # Save datasets for later use
    np.save(os.path.join(save_directory, "train.npy"), X_train)
    np.save(os.path.join(save_directory, "test.npy"), X_test)
    np.save(os.path.join(save_directory, "val.npy"), X_val)
    np.save(os.path.join(save_directory, "train_labels.npy"), y_train)
    np.save(os.path.join(save_directory, "test_labels.npy"), y_test)
    np.save(os.path.join(save_directory, "val_labels.npy"), y_val)

    print(f" Preprocessing complete. Data saved in '{save_directory}/'")
    print(f"🔹 Train shape: {X_train.shape}, Test shape: {X_test.shape}, Validation shape: {X_val.shape}")
    print("🔹 Min value:", np.min(X_train))
    print("🔹 Max value:", np.max(X_train))
    print("🔹 Training set - Malware count:", np.sum(y_train == 1))
    print("🔹 Training set - Safe count:", np.sum(y_train == 0))
    print("🔹 Test set - Malware count:", np.sum(y_test == 1))
    print("🔹 Test set - Safe count:", np.sum(y_test == 0))
    print("🔹 Validation set - Malware count:", np.sum(y_val == 1))
    print("🔹 Validation set - Safe count:", np.sum(y_val == 0))

    # Call the new function to display a sample of the normalized data
    #display_normalized_data(X_train, X_test, X_val)
