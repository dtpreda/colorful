import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from colour.soft_encode import soft_encode

def reweight(ab, weights):
    ab_max = np.argmax(ab, axis=-1)

    weights = weights[ab_max]

    return weights

if __name__ == "__main__":
    matplotlib.use("WX")
    weights = np.load("data/class_rebalance_weights.npy")
    hull = np.load("data/hull.npy")
    ab = np.array([[[[50, -51], [-52,50]], [[50, -51], [-52,50]]], [[[100, -1], [-52,50]], [[50, -51], [-52,50]]], [[[50, -51], [-52,50]], [[50, -51], [-52,50]]]], dtype=np.float32)
    ab = soft_encode(ab, centroids=hull, n=5)
    weights = reweight(ab, weights)

    print(weights.shape)
    print(weights)

