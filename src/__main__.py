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

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--dataroot", type=str, default="./data")

def set_device(gpu):
    use_gpu = args.gpu != -1 and torch.cuda.is_available()
    if use_gpu:
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
    
    device = torch.device(args.gpu if use_gpu else "cpu")

    return device

def get_dataloader(dataroot, batch_size, num_workers, split='unlabeled'):
    def import_image(img):
        return torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img)), (2,0,1)))
    
    transform = transforms.Compose([
        transforms.Lambda(import_image),
    ])

    dataset = torchvision.datasets.STL10(root=dataroot, split=split,
                                            download=True, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)
    
    return dataloader

if __name__ == "__main__":
    args = parser.parse_args()

    device = set_device(args.gpu)
    
    hull = np.load("data/hull.npy")
    weights = np.load("data/stl10/class_rebalance_weights.npy")

    trainloader = get_dataloader(args.dataroot, args.batch_size, args.num_workers, split='unlabeled')
    testloader = get_dataloader(args.dataroot, args.batch_size, args.num_workers, split='test')
    
    TRAIN_SIZE = len(trainloader)
    TEST_SIZE = len(testloader)

    model = Colorizer()
    model.train()
    model.to(device)

    epochs = args.epochs
    learning_rate = args.learning_rate

    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_loss = []
    test_loss = []

    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        with tqdm(total=TRAIN_SIZE) as pbar:
            for i, data in enumerate(trainloader):
                inputs, _ = data
                inputs = inputs.to(device)
                l, ab = inputs[:, 0, :, :], inputs[:, 1:, :, :]
                l = l.unsqueeze(1)
                prediction = model(l)
                prediction = prediction.permute(0, 2, 3, 1)
                ab = ab.permute(0, 2, 3, 1)

                ground_truth = soft_encode(ab, centroids=hull, n=5)
                pixelwise_weights = reweight(ground_truth, weights)

                loss = -torch.sum(pixelwise_weights * torch.sum(ground_truth * torch.log(prediction), dim=-1), dim=(-1, -2))
                loss = torch.mean(loss)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                train_loss.append(loss.item())

                pbar.update(1)
            
            pbar.close()

        with tqdm(total=TEST_SIZE) as pbar:
            print("Testing Epoch {}".format(epoch))
            for i, data in enumerate(testloader):
                inputs, _ = data
                inputs = inputs.to(device)
                l, ab = inputs[:, 0, :, :], inputs[:, 1:, :, :]
                l = l.unsqueeze(1)
                prediction = model(l)
                prediction = prediction.permute(0, 2, 3, 1)
                ab = ab.permute(0, 2, 3, 1)

                ground_truth = soft_encode(ab.cpu().numpy(), centroids=hull, n=5)
                pixelwise_weights = reweight(ground_truth, weights)

                # prediction = torch.FloatTensor(prediction)
                ground_truth = torch.FloatTensor(ground_truth).to(device)
                pixelwise_weights = torch.FloatTensor(pixelwise_weights).to(device)

                loss = -torch.sum(pixelwise_weights * torch.sum(ground_truth * torch.log(prediction), dim=-1), dim=(-1, -2))
                loss = torch.mean(loss)

                test_loss.append(loss.item())

                pbar.update(1)
            
            pbar.close()
        
        print("Epoch {} train loss: {}".format(epoch, np.mean(train_loss[-TRAIN_SIZE:])))
        print("Epoch {} test loss: {}".format(epoch, np.mean(test_loss[-TEST_SIZE:])))
        
        if epoch != 0 and np.mean(test_loss[-TEST_SIZE:]) < np.mean(test_loss[-TEST_SIZE-TEST_SIZE:-TEST_SIZE]):
            print("Saving model at epoch {}".format(epoch))
            torch.save(model.state_dict(), "model_best.pth")

    
    torch.save(model.state_dict(), "model_latest.pth")

    # plot losses
    matplotlib.use('Qt5Agg')
    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.savefig("loss.png")
    