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
        distances[i] = mnist_train[i][1], distance
        
    sorted_distances = np.argsort(distances)
    k_elements = sorted_distances[:choose_k]
    return torch.mode(k_elements).values.item(0)

find_match(np.array(mnist_test[1]), 5)
