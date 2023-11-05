import random
import cv2
import numpy as np
import torchvision
import torch
import PIL

mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)


def find_match(arr, choose_k):
    distances = np.empty(1000, dtype=object) 
    random_indices = random.sample(range(len(mnist_train)), 1000)
    for i in random_indices:
        sample_image, label = mnist_train[i]
        sample_image_arr = np.array(sample_image, dtype=np.int32)
        distance = np.sqrt(np.sum((arr - sample_image_arr) ** 2))
        distances[i] = distance
        
    sorted_distances = np.argsort(distances)
    k_indices = sorted_distances[:choose_k]
    predicted_labels = [mnist_train[i][1] for i in k_indices]

    return torch.mode(predicted_labels).values.item(0)

num_trials = 500
actual_labels = np.empty(1000, dtype=object)
predicted_labels = np.empty(1000, dtype=object)

for i in range(num_trials):
    input_image, true_label = mnist_test[i]
    predicted_label = find_match(np.array(input_image).reshape(-1), 5)
    actual_labels[i] = true_label
    predicted_labels[i] = predicted_label

accuracy = (sum(actual_labels[i] == predicted_labels[i] for i in range(500)) / 500) * 100
print("Accuracy: {:.2f}%".format(accuracy))
