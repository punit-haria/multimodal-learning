from data import MNIST, CIFAR
from training import Results
from results import image_plot


from runcolor import models


experiment_name = "ar_test"
suffix = None  # None

data = MNIST()    # MNIST, CIFAR


tracker = Results.load(experiment_name)  # performance tracker

syntheses = ['reconstruct', 'sample', 'fix_latents', 'latent_activations']

image_plot(tracker, models, data=data, suffix=suffix, syntheses=syntheses,
           n_rows=8, n_cols=8, n_pixels=0, spacing=0)
