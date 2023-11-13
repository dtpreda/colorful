import torch
from src.net.arch import Colorizer
from src.net.annealed import annealed_mean

import numpy as np
import torchvision
import torchvision.transforms as transforms
from skimage import color
from argparse import ArgumentParser
import matplotlib.pyplot as plt

def import_image(img):
    return torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img)), (2,0,1)))

parser = ArgumentParser()
parser.add_argument("--batch-size", type=int, default=10)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--model", type=str, default="model.pth")


if __name__ == "__main__":
    args = parser.parse_args()
    use_gpu = args.gpu != -1 and torch.cuda.is_available()
    if use_gpu:
        print("There are ", torch.cuda.device_count(), " available GPUs!")
        torch.cuda.set_device(args.gpu)
        print('Current single GPU: {}'.format(torch.cuda.current_device()))

    state_dict = torch.load(args.model)
    
    transform = transforms.Compose([
        transforms.Lambda(import_image),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers)
    
    device = torch.device(args.gpu if use_gpu else "cpu")

    model = Colorizer()
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        inputs = inputs.to(device)
        l, ab = inputs[:, 0, :, :], inputs[:, 1:, :, :]
        l = l.unsqueeze(1)
        prediction = model(l)
        prediction = prediction.permute(0, 2, 3, 1)
        ab = ab.permute(0, 2, 3, 1)

        imgs = annealed_mean(prediction.cpu().detach().numpy())
        
        first_img_predicted = [l[0].cpu().detach().numpy(), imgs[0]]
        first_img_true = [l[0].cpu().detach().numpy(), ab[0]]
        
        # plot side by side
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(color.lab2rgb(np.transpose(first_img_predicted, (1,2,0))))
        axs[1].imshow(color.lab2rgb(np.transpose(first_img_true, (1,2,0))))
        plt.show()
        break

    
        
