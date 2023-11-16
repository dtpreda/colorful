import matplotlib
import matplotlib.pyplot as plt

from src.net.arch import Colorizer
from src.colour.soft_encode import soft_encode
from src.weight import reweight

import torchvision.transforms as transforms
from skimage import color
from tqdm import tqdm
import torchvision
import numpy as np
import torch
from torch.optim import Adam
from torch import cuda
import os
from datetime import datetime

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning-rate", type=float, default=3e-5)
parser.add_argument("--weight-decay", type=float, default=1e-3)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--dataroot", type=str, default="./data")
parser.add_argument("--dataset", type=str, default="stl10", choices=["stl10", "cifar-10"])
parser.add_argument("--checkpoint_dir", type=str, default="./out")

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

    checkpoint_dir = os.path.join(args.checkpoint_dir, args.dataset + "_" + datetime.now().strftime("%b%dT%H-%M") + "_" + str(hash(datetime.now().microsecond)))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    hull = torch.from_numpy(np.load("data/hull.npy")).to(device)
    weights = torch.from_numpy(np.load(f"data/{args.dataset}/class_rebalance_weights.npy")).to(device)

    trainloader, testloader = get_dataloaders(args.dataset, args.dataroot, args.batch_size, args.num_workers)
    
    TRAIN_SIZE = len(trainloader)
    TEST_SIZE = len(testloader)

    model = Colorizer()
    model.train()
    model.to(device)

    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        model.train()
        train_loss.append(0)
        with tqdm(total=TRAIN_SIZE) as pbar:
            for i, data in enumerate(trainloader):
                inputs, _ = data
                inputs = inputs.to(device)
                l, ab = inputs[:, 0, :, :], inputs[:, 1:, :, :]
                l = l.unsqueeze(1)
                prediction = model(l)
                prediction = prediction.permute(0, 2, 3, 1)
                ab = ab.permute(0, 2, 3, 1)

                ground_truth = soft_encode(ab, centroids=hull)
                pixelwise_weights = reweight(ground_truth, weights)

                loss = -torch.sum(pixelwise_weights * torch.sum(ground_truth * torch.log(prediction + 1e-8), dim=-1), dim=(-1, -2))
                loss = torch.mean(loss)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                # train_loss.append(loss.item())
                train_loss[-1] += loss.item()
                pbar.update(1)
            
            pbar.close()

        train_loss[-1] /= TRAIN_SIZE
        print("Epoch {} train loss: {}".format(epoch, train_loss[-1]))
        
        model.eval()
        with tqdm(total=TEST_SIZE) as pbar:
            print("Testing Epoch {}".format(epoch))
            test_loss.append(0)
            for i, data in enumerate(testloader):

                with torch.no_grad():
                    inputs, _ = data
                    inputs = inputs.to(device)
                    l, ab = inputs[:, 0, :, :], inputs[:, 1:, :, :]
                    l = l.unsqueeze(1)
                    prediction = model(l)
                    prediction = prediction.permute(0, 2, 3, 1)
                    ab = ab.permute(0, 2, 3, 1)

                    ground_truth = soft_encode(ab, centroids=hull)
                    pixelwise_weights = reweight(ground_truth, weights)

                    loss = -torch.sum(pixelwise_weights * torch.sum(ground_truth * torch.log(prediction + 1e-8), dim=-1), dim=(-1, -2))
                    loss = torch.mean(loss)

                    # test_loss.append(loss.item())
                    test_loss[-1] += loss.item()

                    pbar.update(1)
            
            pbar.close()
        
        test_loss[-1] /= TEST_SIZE
        print("Epoch {} test loss: {}".format(epoch, test_loss[-1]))
        
        if epoch == 0 or test_loss[-1] < min(test_loss[:-1]):
            print("Saving model at epoch {}".format(epoch))
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_best.pth".format(epoch)))

    
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_latest.pth"))

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.savefig(os.path.join(checkpoint_dir, "loss.png"))
    