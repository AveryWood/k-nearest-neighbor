# MNIST k-Nearest Neighbors Classifier

This project demonstrates a basic **k-Nearest Neighbors (k-NN)** classification algorithm on the **MNIST** dataset of handwritten digits. The code is written in Python and uses the **PyTorch** library for loading the MNIST dataset and performing the classification task.

## Description

This project implements a simple k-NN classifier that predicts handwritten digit labels from the **MNIST** dataset. The script computes the Euclidean distance between a test image and random samples from the training set, then predicts the most common label among the nearest neighbors.

### Features:
- Uses **PyTorch**'s `torchvision.datasets.MNIST` to automatically download and load the MNIST dataset.
- Implements a **k-Nearest Neighbors (k-NN)** algorithm to classify test images based on their similarity to randomly chosen training images.
- Outputs the **accuracy** of the model after classifying 500 test samples.

## Requirements

To run the code, you will need to have the following Python packages installed:

- **torch** (for PyTorch)
- **torchvision** (for MNIST dataset)
- **numpy** (for numerical operations)

