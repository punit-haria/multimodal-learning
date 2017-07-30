from data import MNIST
from training import Results
from results import image_plot


from runvae import models, experiment_name



tracker = Results.load(experiment_name)  # performance tracker
mnist = MNIST()  # data



#image_plot(tracker, models, data=mnist, suffix=suffix, n_rows=8, n_cols=8, n_pixels=300,
#           spacing=0, synthesis_type='fix_latents')

image_plot(tracker, models, data=mnist, suffix=None, n_rows=8, n_cols=8, n_pixels=300,
           spacing=0, synthesis_type='reconstruct')

image_plot(tracker, models, data=mnist, suffix=None, n_rows=8, n_cols=8, n_pixels=300,
           spacing=0, synthesis_type='sample')

image_plot(tracker, models, data=mnist, suffix=None, n_rows=8, n_cols=8, n_pixels=300,
           spacing=0, synthesis_type='latent_activations')


