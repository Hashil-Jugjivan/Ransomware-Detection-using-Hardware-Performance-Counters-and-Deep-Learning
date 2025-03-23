# Ransomware Detection using Hardware Performance Counters and Deep Learning

This repository contains the implementation of various deep learning architectures for ransomware detection using Hardware Performance Counters (HPCs). The project explores how microarchitectural events captured by HPCs can be leveraged with advanced deep learning techniques to effectively detect ransomware during its pre-encryption phase.

## Project Overview

Ransomware has emerged as a significant cybersecurity threat with global damages projected to reach $265 billion annually by 2031. Traditional detection methods face limitations including ineffectiveness against zero-day attacks, high system overheads, and vulnerability to evasion techniques.

This project proposes an innovative approach using Hardware Performance Counters (HPCs) combined with optimized deep learning architectures to overcome these limitations. The implementation includes both manually designed and Neural Architecture Search (NAS) optimized models.

## Approach

The project follows these key steps:

1. Collection and preprocessing of HPC traces from ransomware and benign applications
2. Implementation of various deep learning architectures
3. Optimization of architectures using Neural Architecture Search (NAS)
4. Comprehensive evaluation and comparison with state-of-the-art frameworks

## Models Implemented

Seven deep learning models were developed and evaluated:

1. **CNN**: A convolutional neural network for spatial feature extraction
2. **Hybrid CNN-RNN**: Combines CNN's spatial features with RNN's temporal processing
3. **LSTM**: Captures temporal dependencies in HPC data
4. **Transformer**: Leverages self-attention mechanisms for parallel processing
5. **NAS-CNN**: Optimized CNN architecture discovered through NAS
6. **NAS-LSTM**: Optimized LSTM architecture discovered through NAS
7. **NAS-Transformer**: Optimized Transformer architecture discovered through NAS

## Repository Structure

- **CNN/**: Implementation of the CNN model, including training and evaluation scripts
- **CNN_NAS/**: Implementation of the NAS-optimized CNN model
- **CNN_RNN/**: Implementation of the hybrid CNN-RNN model
- **LSTM/**: Implementation of the LSTM model
- **LSTM_NAS/**: Implementation of the NAS-optimized LSTM model
- **Transformer/**: Implementation of the Transformer model
- **Transformer_NAS/**: Implementation of the NAS-optimized Transformer model
- **Data_Preprocessing/**: Scripts for data preprocessing and normalization
  - **normalization_small.py**: Implementation of data normalization with train/val/test splits
  - **norm_small_noval.py**: Implementation of data normalization with train/test splits
- **Dataset/**: Directory for storing the dataset
- **Models/**: Directory for storing trained models

## Key Features

- **Hardware Performance Counters**: Utilizes 23 different HPC metrics to capture unique ransomware behavior patterns
- **Neural Architecture Search**: Automated optimization of model architectures for improved performance
- **Comparison with SOTA**: Benchmarking against state-of-the-art ransomware detection frameworks
- **Multiple Model Architectures**: Exploration of different deep learning approaches for the task

## Results

The models achieved impressive performance:

| Model | Accuracy | Precision | Recall | F1 Score | Parameters | MCC |
|-------|----------|-----------|--------|----------|------------|-----|
| CNN | 97.41% | 0.98 | 0.97 | 0.97 | 288,449 | 0.950 |
| Hybrid CNN-RNN | 97.41% | 0.9742 | 0.97 | 0.97 | 1,671,234 | 0.948 |
| LSTM | 98.28% | 0.9661 | 1.00 | 0.98 | 56,129 | 0.966 |
| Transformer | 98.28% | 1.00 | 0.9649 | 0.98 | 139,361 | 0.966 |
| NAS-CNN | 99.14% | 0.99 | 0.99 | 0.99 | 118,977 | 0.983 |
| NAS-LSTM | 99.14% | 1.00 | 0.9825 | 0.99 | 16,769 | 0.983 |
| NAS-Transformer | 98.28% | 1.00 | 0.9649 | 0.98 | 115,361 | 0.966 |

The NAS-optimized LSTM model emerged as the most efficient solution, achieving 99.14% accuracy with only 16,769 parameters. This outperforms state-of-the-art frameworks including HiPeR (98.68%) and DeepWare (98.6%).

## Requirements

```
Python 3.11.3
PyTorch
TensorFlow
Optuna
Scikit-learn
Matplotlib
NumPy
Pandas
```

See `requirements.txt` for a complete list of dependencies.

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/Hashil-Jugjivan/Ransomware-Detection-using-Hardware-Performance-Counters-and-Deep-Learning.git
```
2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate # On Windows, use: venv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**:
   ```python
   python Data_Preprocessing/normalization_small.py
   ```
2. **Training Models**:
   ```python
   # Train CNN
   python CNN/train_cnn.py
   
   # Train CNN-RNN
   python CNN_RNN/train_cnn_rnn.py
   
   # Train LSTM
   python LSTM/train_lstm.py
   
   # Train Transformer
   python Transformer/train_transformer.py
   ```

3. **Neural Architecture Search**:
   ```python
   # CNN NAS
   python CNN_NAS/cnn_nas.py
   
   # LSTM NAS
   python LSTM_NAS/lstm_nas.py
   
   # Transformer NAS
   python Transformer_NAS/transformer_nas.py
   ```

4. **Evaluation**:
   ```python
   python CNN/evaluate_cnn.py
   ```

## Future Work

- Expanding the dataset to include a wider variety of ransomware families and benign applications
- Investigating ensemble methods and more sophisticated architectures
- Integrating the models into real-time detection systems
- Exploring multi-class classification for specific ransomware family identification

## Citation

If you use this code in your research, please cite:

Jugjivan, H. (2024). Detecting Ransomware using Deep Learning and Hardware Performance Counters.
Nanyang Technological University, Singapore.


