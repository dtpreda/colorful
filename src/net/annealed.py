import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch

def annealed_mean(z, T=0.38):
    q = torch.exp(torch.log(z + 1e-8) / T)
    return q / (torch.sum(q, axis=-1, keepdims=True) + 1e-8)

def z_to_y(z, hull):
    mean = annealed_mean(z)
    expected_value = torch.sum(mean * torch.arange(hull.shape[0]).to(z.device), dim=-1, keepdim=True)
    expected_value = torch.round(expected_value).to(torch.int64)
    return hull[expected_value].squeeze(-2)

if __name__ == "__main__":
    distribution = np.load("data/stl10/empirical_distribution.npy")
    hull = np.load("data/hull.npy")
    # mean = annealed_mean(np.array([distribution, distribution], dtype=np.float64))
    # mean = mean[0]

    for i in range(100, 0, -1):
        t = (i / 100)
        mean, q = annealed_mean(np.array([distribution], dtype=np.float64))
        print(f"For T={t}, expected value={np.sum(mean * np.arange(hull.shape[0]), axis=-1, keepdims=True)}")
        print(f"Q sum = Z")
        print("-"*15)
    # plt.plot(distribution)
    # plt.plot(mean, color='r', linestyle='--')
    # plt.savefig("annealead_mean.png")