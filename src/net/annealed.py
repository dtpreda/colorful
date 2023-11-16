import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def annealed_mean(z, T=0.20):
    # if z is not a numpy array, convert it
    if not isinstance(z, np.ndarray):
        z = np.array(z, dtype=np.float64)

    q = np.exp(np.log(z + 1e-8) / T)
    return q / (np.sum(q, axis=-1, keepdims=True) + 1e-8)

def z_to_y(z):
    hull = np.load("data/hull.npy")
    mean = annealed_mean(z)
    expected_value = np.sum(mean * np.arange(hull.shape[0]), axis=-1, keepdims=True)
    expected_value = np.round(expected_value).astype(np.int64)
    print(expected_value.shape)
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