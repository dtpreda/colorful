import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
from skimage import color
from tqdm import tqdm
import torchvision
import numpy as np
import torch
matplotlib.use('Agg')

from colour.soft_encode import soft_encode

BATCH_SIZE = 64
UNIFORM_LAMBDA = 0.5
GAUSSIAN_SIGMA = 5

def import_image(img):
    return torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img)), (2,0,1)))

transform = transforms.Compose([
    transforms.Lambda(import_image),
])

# Load the CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=4)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)

TRAIN_SIZE = len(trainloader)

def to_discrete(img):
    hull = np.load("data/hull.npy")
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.float32)
    encoding = soft_encode(img, centroids=hull, n=5)
    return encoding

if __name__ == "__main__":
    distribution = np.zeros((32, 32, 326,), dtype=np.float32)

    with tqdm(total=TRAIN_SIZE) as pbar:
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = np.transpose(inputs.numpy(), (0,2,3,1))
            l, ab = inputs[:,:,:,0], inputs[:,:,:,1:]
            encoding = to_discrete(ab)

            distribution += np.sum(encoding, axis=(0)) / (TRAIN_SIZE * BATCH_SIZE)

            pbar.update(1)
    
    distribution = np.sum(distribution, axis=(0,1)) / (32 * 32)
    distribution = gaussian_filter(distribution, sigma=GAUSSIAN_SIGMA)
    distribution = distribution / np.sum(distribution)

    weights = (1 - UNIFORM_LAMBDA) * distribution + UNIFORM_LAMBDA / 326
    weights = 1 / weights

    product_distribution = weights * distribution
    expected_value = np.sum(product_distribution) / np.sum(distribution)
    weights = weights / expected_value

    np.save("data/empirical_distribution.npy", distribution)
    np.save("data/class_rebalance_weights.npy", weights)

