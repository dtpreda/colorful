import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter
from skimage import color
from tqdm import tqdm
import torchvision
import numpy as np
import torch
from argparse import ArgumentParser

from src.colour.soft_encode import soft_encode

parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--uniform-lambda", type=float, default=0.5)
parser.add_argument("--gaussian-sigma", type=float, default=5)
parser.add_argument("--dataroot", type=str, default="./data")

def to_discrete(img):
    hull = np.load("data/hull.npy")
    if not isinstance(img, np.ndarray):
        img = np.array(img, dtype=np.float32)
    encoding = soft_encode(img, centroids=hull, n=5)
    return encoding

if __name__ == "__main__":
    args = parser.parse_args()

    def import_image(img):
        return torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img)), (2,0,1)))

    transform = transforms.Compose([
        transforms.Lambda(import_image),
    ])

    # Load the STL10 dataset
    trainset = torchvision.datasets.STL10(root=args.dataroot, split='unlabeled',
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers)

    TRAIN_SIZE = len(trainloader)

    distribution = np.zeros((96, 96, 326,), dtype=np.float32)

    with tqdm(total=TRAIN_SIZE) as pbar:
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = np.transpose(inputs.numpy(), (0,2,3,1))
            l, ab = inputs[:,:,:,0], inputs[:,:,:,1:]
            encoding = to_discrete(ab)

            distribution += np.sum(encoding, axis=(0)) / (TRAIN_SIZE * args.batch_size)

            pbar.update(1)
    
    distribution = np.sum(distribution, axis=(0,1)) / (32 * 32)
    distribution = gaussian_filter(distribution, sigma=args.gaussian_sigma)
    distribution = distribution / np.sum(distribution)

    weights = (1 - args.uniform_lambda) * distribution + args.uniform_lambda / 326
    weights = 1 / weights

    product_distribution = weights * distribution
    expected_value = np.sum(product_distribution) / np.sum(distribution)
    weights = weights / expected_value

    np.save("data/empirical_distribution.npy", distribution)
    np.save("data/class_rebalance_weights.npy", weights)

