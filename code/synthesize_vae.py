from data import MNIST, CIFAR
from training import Results
from results import image_plot


from runcolor import models, experiment_name



tracker = Results.load(experiment_name)  # performance tracker


data = CIFAR()    # MNIST, CIFAR


syntheses = ['reconstruct', 'sample', 'fix_latents', 'latent_activations']

image_plot(tracker, models, data=data, suffix=None, syntheses=syntheses,
           n_rows=8, n_cols=8, n_pixels=300, spacing=0, )
