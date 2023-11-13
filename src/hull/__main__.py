import numpy as np
from src.colour.quantize import in_hull_centroids

if __name__ == "__main__":
    hull = in_hull_centroids()
    np.save("data/hull.npy", hull)