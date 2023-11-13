import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from colour.soft_encode import soft_encode
import torch

def reweight(ab, weights):
    ab_max = torch.argmax(ab, axis=-1)
    weights = weights[ab_max]

    return weights

if __name__ == "__main__":
    matplotlib.use("WX")
    
    weights = torch.from_numpy(np.load("data/cifar-10/class_rebalance_weights.npy"))
    hull = torch.from_numpy(np.load("data/hull.npy"))
    
    ab = torch.tensor([[[[50, -51], [-52,50]], [[50, -51], [-52,50]]], [[[100, -1], [-52,50]], [[50, -51], [-52,50]]], [[[50, -51], [-52,50]], [[50, -51], [-52,50]]]], dtype=torch.float32)
    
    ab = soft_encode(ab, centroids=hull, n=5)
    weights = reweight(ab, weights)

    print(weights.shape)
    print(weights)

