from __future__ import print_function
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
import os
import numpy as np
from scipy.spatial.distance import cosine
import random

def main():
    print('Loading MNIST')
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('~/shared', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True
    )
    print('MNIST has been loaded.')

if __name__ == '__main__':
    main()