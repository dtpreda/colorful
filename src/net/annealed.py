import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def annealed_mean(z, T=0.38):
    # if z is not a numpy array, convert it
    if not isinstance(z, np.ndarray):
        z = np.array(z, dtype=np.float32)

    f = np.exp(np.log(z) / T) / (np.sum(np.exp(np.log(z) / T), axis=-1, keepdims=True) + 1e-8)

    return np.sum(f * np.arange(326), axis=-1, keepdims=True)

def mean_to_ab(mean, hull):
    mean = np.round(mean).astype(np.int64)
    return hull[mean]

def z_to_y(z):
    hull = np.load("data/hull.npy")
    mean = annealed_mean(z + 1e-8)
    return mean_to_ab(mean, hull)

if __name__ == "__main__":
    distribution = np.load("data/empirical_distribution.npy")
    hull = np.load("data/hull.npy")
    # get mean of probability distribution
    mean = annealed_mean(np.array([distribution, distribution], dtype=np.float32))
    mean = mean[0]

    matplotlib.use('Qt5Agg')

    # plot as line graph
    plt.plot(distribution)
    # plot mean as vertical line
    plt.axvline(x=mean, color='r', linestyle='--')
    plt.show()

    # convert mean to ab
    ab = mean_to_ab(mean, hull)
    print(ab)