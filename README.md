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

## Results

The accuracy of the k-NN model on 500 randomly selected test images will be printed to the console. You can experiment with different values of `k` to see how it affects the accuracy.

## Limitations

- This implementation uses a simple k-NN approach and is not optimized for performance.
- The model only uses 1,000 random samples from the training set for each test image, which can limit its accuracy.
- It may be slow for larger datasets as it does not utilize efficient indexing methods for nearest neighbor searches.

## Acknowledgements

- **MNIST dataset**: A widely-used dataset for training and testing image classification models.
- **PyTorch**: For providing easy-to-use functions for dataset handling and tensor operations.
