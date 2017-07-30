from data import MNIST, CIFAR
from training import Results
from results import image_plot


from runcolor import models, experiment_name



tracker = Results.load(experiment_name)  # performance tracker


data = CIFAR()    # MNIST, CIFAR



image_plot(tracker, models, data=data, suffix=None, n_rows=8, n_cols=8, n_pixels=300,
           spacing=0, synthesis_type='fix_latents')

image_plot(tracker, models, data=data, suffix=None, n_rows=8, n_cols=8, n_pixels=300,
           spacing=0, synthesis_type='reconstruct')

image_plot(tracker, models, data=data, suffix=None, n_rows=8, n_cols=8, n_pixels=300,
           spacing=0, synthesis_type='sample')

image_plot(tracker, models, data=data, suffix=None, n_rows=8, n_cols=8, n_pixels=300,
           spacing=0, synthesis_type='latent_activations')


