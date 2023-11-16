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
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--dataset", type=str, default="stl10", choices=["stl10", "cifar-10"])

def set_device(gpu):
    use_gpu = args.gpu != -1 and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
    
    device = torch.device(args.gpu if use_gpu else "cpu")

    return device

def get_dataloaders(dataset, dataroot, batch_size, num_workers):
    def import_image(img):
        return torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img)), (2,0,1)))
    
    transform = transforms.Compose([
        transforms.Lambda(import_image),
    ])

    if dataset == "stl10":
        traindataset = torchvision.datasets.STL10(root=dataroot, split='unlabeled',
                                                download=True, transform=transform)
        testdataset = torchvision.datasets.STL10(root=dataroot, split='test',
                                                download=True, transform=transform)
    elif dataset == "cifar-10":
        traindataset = torchvision.datasets.CIFAR10(root=dataroot, train=True,
                                                download=True, transform=transform)
        testdataset = torchvision.datasets.CIFAR10(root=dataroot, train=False,
                                                download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    
    return trainloader, testloader

if __name__ == "__main__":
    args = parser.parse_args()

    device = set_device(args.gpu)
    
    hull = torch.from_numpy(np.load("data/hull.npy")).to(device)

    trainloader, _ = get_dataloaders(args.dataset, args.dataroot, args.batch_size, args.num_workers)

    TRAIN_SIZE = len(trainloader)

    distribution = torch.zeros(trainloader.dataset[0][0].shape[1], trainloader.dataset[0][0].shape[2], 326).to(device)

    with tqdm(total=TRAIN_SIZE) as pbar:
        for i, data in enumerate(trainloader):
            inputs, _ = data
            inputs = inputs.to(device)
            inputs = inputs.permute(0, 2, 3, 1)
            l, ab = inputs[:,:,:,0], inputs[:,:,:,1:]
            encoding = soft_encode(ab, centroids=hull)

            distribution += torch.sum(encoding, axis=(0))

            pbar.update(1)
    
    distribution = distribution.cpu().numpy()
    distribution = distribution / (TRAIN_SIZE * args.batch_size)
    distribution = np.sum(distribution, axis=(0,1)) / (trainloader.dataset[0][0].shape[1] * trainloader.dataset[0][0].shape[2])
    distribution = gaussian_filter(distribution, sigma=args.gaussian_sigma)
    distribution = distribution / np.sum(distribution)

    weights = (1 - args.uniform_lambda) * distribution + args.uniform_lambda / 326
    weights = 1 / weights

    weights = weights / np.sum(weights * distribution)

    print(f"Distribution sum: {np.sum(distribution)} mean: {np.mean(distribution)} std: {np.std(distribution)}")
    print(f"Weights sum: {np.sum(weights)} mean: {np.mean(weights)} std: {np.std(weights)}")
    print(f"Product sum: {np.sum(weights * distribution)} mean: {np.mean(weights * distribution)} std: {np.std(weights * distribution)}")
    
    np.save(f"data/{args.dataset}/empirical_distribution.npy", distribution)
    np.save(f"data/{args.dataset}/class_rebalance_weights.npy", weights)

