import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from colour.soft_encode import soft_encode

def reweight(ab, weights):
    ab_max = np.argmax(ab, axis=-1)

    weights = weights[ab_max]

    rebalanced = ab * weights[..., None]
    return rebalanced

if __name__ == "__main__":
    matplotlib.use("WX")
    weights = np.load("data/class_rebalance_weights.npy")
    hull = np.load("data/hull.npy")
    ab = np.array([[[[50, -51], [-52,50]], [[50, -51], [-52,50]]], [[[100, -1], [-52,50]], [[50, -51], [-52,50]]], [[[50, -51], [-52,50]], [[50, -51], [-52,50]]]], dtype=np.float32)
    ab = soft_encode(ab, centroids=hull, n=5)
    ab_after = reweight(ab, weights)

    pixel_before = ab[0, 0, 0, :]
    pixel_after = ab_after[0, 0, 0, :]

    plt.plot(pixel_before, label="before")
    plt.plot(pixel_after, label="after")
    plt.xlim([45, 80])
    plt.legend()
    plt.show()