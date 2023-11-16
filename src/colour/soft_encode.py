from sklearn.neighbors import NearestNeighbors
# from src.colour.quantize import get_ab_centroids
import numpy as np

import torch

def soft_encode(ab, centroids, neighbours=5):
    n, h, w, c = ab.shape
    ab = ab.reshape(-1, 2)
    distances = torch.cdist(ab, centroids)
    distances, indices = torch.topk(distances, k=neighbours, dim=-1, largest=False, sorted=True)
    distances = torch.exp(-distances**2 / (2 * 5**2))
    distances = distances / torch.sum(distances, dim=-1, keepdim=True)

    encoding = torch.zeros((ab.shape[0], centroids.shape[0]), dtype=torch.float32).to(ab.device)
    encoding[torch.arange(ab.shape[0])[:, None], indices] = distances

    encoding = encoding.reshape(n, h, w, -1)

    return encoding

if __name__ == "__main__":
    hull = torch.from_numpy(np.load("data/hull.npy"))
    ab = np.array([[[[50, -51], [-52,50]], [[50, -51], [-52,50]]], [[[50, -51], [-52,50]], [[50, -51], [-52,50]]], [[[50, -51], [-52,50]], [[50, -51], [-52,50]]]], dtype=np.float32)
    ab = torch.from_numpy(ab)
    encoding = soft_encode(ab, hull, neighbours=5)
    print(encoding.shape)
    print(encoding[0, 0, 0, :])