# MNIST Digit Classification with CNN (PyTorch)

This repository contains a high-performance **Convolutional Neural Network (CNN)** implementation designed to classify handwritten digits from the MNIST dataset. The project demonstrates advanced deep learning practices, including modular code structure, data normalization, and hyperparameter optimization.

## 🎯 Project Objectives
* Implement a robust CNN architecture using **PyTorch**.
* Achieve high classification accuracy (98%+) on the MNIST test set.
* Demonstrate professional machine learning workflows: Data augmentation, normalization, and model evaluation.

## 🏗️ Model Architecture
The network consists of a series of convolutional layers followed by fully connected layers to extract spatial features effectively:
* **Convolutional Layer 1:** 32 filters, 3x3 kernel, ReLU activation.
* **Max Pooling:** 2x2 window for spatial dimension reduction.
* **Convolutional Layer 2:** 64 filters, 3x3 kernel, ReLU activation.
* **Fully Connected Layer 1:** 128 neurons with ReLU activation.
* **Output Layer:** 10 neurons with Log-Softmax (representing digits 0-9).

## 🚀 Key Features
* **Modular Design:** Separate files for model definition, training, and utilities for better maintainability.
* **Data Preprocessing:** Implemented Mean/STD normalization to speed up convergence.
* **Performance Monitoring:** Integrated loss curve plotting and visual prediction analysis using Matplotlib.
* **GPU Support:** Automatic device detection (CUDA/MPS/CPU) for hardware acceleration.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** PyTorch
* **Libraries:** Torchvision, NumPy, Matplotlib

## 📂 Project Structure
```text
MNIST-CNN-PyTorch/
├── src/
│   ├── model.py        # CNN Architecture
│   ├── train.py        # Training logic
│   └── utils.py        # Visualization tools
├── main.py             # Entry point
├── requirements.txt    # Dependencies
└── README.md           # Documentation
