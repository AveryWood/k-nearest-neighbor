import random
import cv2
import numpy as np
import torchvision
import torch
import PIL

mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True)


def find_match(arr,choose_k):
    distances = np.empty(1000, dtype=object) 
    matches = np.empty(1000, dtype=object)
    random_indices = random.sample(range(len(mnist_test)), 1000)
    for i in random_indices:
        sample_image, label = mnist_train[i]
        sample_image_arr = np.array(sample_image, dtype=np.int32)
        distance = np.sqrt(sum((arr - sample_image_arr)**2))
        distances[i] = distance
        
        
    sorted_distances = np.argsort(distances)
    k_elements = sorted_distances[:choose_k]
    predicted_label = torch.mode(actual_labels[k_indices]).values.item(0)

    return torch.mode(k_elements).values.item(0)

num_trials = 500
actual_labels   = np.empty(500, dtype=object)
predicted_labels = np.empty(500, dtype=object)

for i in range(num_trials):
    predicted_labels[i] = find_match(np.array(mnist_test[i]), 5)
    actual_label[i] = mnist_test[i][1]

print("Accuracy: " + sum(actual_label == predicted_labels) / 1000 * 100 + "%")
