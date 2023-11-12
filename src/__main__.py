from src.net.arch import Colorizer
from src.net.annealed import annealed_mean
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

def import_image(img):
    return torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img)), (2,0,1)))

if __name__ == "__main__":
    args = parser.parse_args()
    use_gpu = args.gpu != -1 and torch.cuda.is_available()
    if use_gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))
    
    hull = np.load("data/hull.npy")
    weights = np.load("data/class_rebalance_weights.npy")

    transform = transforms.Compose([
        transforms.Lambda(import_image),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers)
    
    TRAIN_SIZE = len(trainloader)

    device = torch.device(args.gpu if use_gpu else "cpu")

    model = Colorizer()
    model.train()
    model.to(device)

    epochs = args.epochs
    # batch_size = args.batch_size
    learning_rate = args.learning_rate

    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_loss = []

    for epoch in range(epochs):
        with tqdm(total=TRAIN_SIZE) as pbar:
            for i, data in enumerate(trainloader):
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

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                train_loss.append(loss.item())

                pbar.update(1)
            
            pbar.close()
        
        print("Epoch {} Loss: {}".format(epoch, np.mean(train_loss)))
    
    torch.save(model.state_dict(), "model.pth")
    