from sklearn.neighbors import NearestNeighbors
from src.colour.quantize import get_ab_centroids
import numpy as np

def soft_encode(ab, centroids=get_ab_centroids(), n=5):
    neigh = NearestNeighbors(n_neighbors=n+1).fit(centroids)
    
    # flatten ab
    n, h, w, c = ab.shape
    ab = ab.reshape(-1, 2)
    distances, indices = neigh.kneighbors(ab)
    distances, indices = distances[:, 1:], indices[:, 1:]
    distances = distances / np.sum(distances, axis=1, keepdims=True)
    
    encoding = np.zeros((ab.shape[0], centroids.shape[0]), dtype=np.float32)
    encoding[np.arange(ab.shape[0])[:, None], indices] = distances

    encoding = encoding.reshape(n, h, w, -1)

    return encoding
    

if __name__ == "__main__":
    hull = np.load("data/hull.npy")
    ab = np.array([[[[50, -51], [-52,50]], [[50, -51], [-52,50]]], [[[50, -51], [-52,50]], [[50, -51], [-52,50]]], [[[50, -51], [-52,50]], [[50, -51], [-52,50]]]], dtype=np.float32)
    encoding = soft_encode(ab, centroids=hull, n=5)
    print(encoding.shape)