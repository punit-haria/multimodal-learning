import numpy as np


def one_hot_encoding(array, n_classes):
    """
    One hot encodes input 1-dimensional array.
    """
    return np.eye(n_classes)[array]


def generate_random(n, dim):
    """
    Generates examples from multivariate normal distribution.

    n: number of samples
    dim: variable dimensionality
    """
    # randomly sampled latents from prior distribution N(0,I)
    return np.random.multivariate_normal(mean=np.zeros(latent_dim),
        cov=np.identity(latent_dim),
        size=n_images)


def generate_uniform(n, w):
    """
    Generates 2-dimensional coordinates uniformly spaced in a grid.

    n: granularity of the grid
    w: width of grid
    """
    # 2D grid of evenly spaced latents
    z1 = np.linspace(-w, w, n)
    z2 = np.linspace(-w, w, n)

    # inverse CDF transform
    Z = []
    for i, x in enumerate(z1):
        for j, y in enumerate(z2):
            Z.append([x,y])
    return np.array(Z)
