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


BATCH_SIZE = 64

def import_image(img):
    return torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img)), (2,0,1)))

if __name__ == "__main__":
    hull = np.load("data/hull.npy")
    weights = np.load("data/class_rebalance_weights.npy")

    transform = transforms.Compose([
        transforms.Lambda(import_image),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=4)
    
    TRAIN_SIZE = len(trainloader)

    model = Colorizer()
    model.train()

    learning_rate = 1e-3
    optimizer = Adam(model.parameters(), lr=learning_rate)
    epochs = 10
    train_loss = []

    for epoch in range(epochs):
        with tqdm(total=TRAIN_SIZE) as pbar:
            for i, data in enumerate(trainloader):
                inputs, _ = data
                l, ab = inputs[:, 0, :, :], inputs[:, 1:, :, :]
                l = l.unsqueeze(1)
                prediction = model(l)
                prediction = prediction.permute(0, 2, 3, 1)
                ab = ab.permute(0, 2, 3, 1)

                ground_truth = soft_encode(ab, centroids=hull, n=5)
                pixelwise_weights = reweight(ground_truth, weights)

                prediction = torch.FloatTensor(prediction)
                ground_truth = torch.FloatTensor(ground_truth)
                pixelwise_weights = torch.FloatTensor(pixelwise_weights)

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
    