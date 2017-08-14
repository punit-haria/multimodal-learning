from data import MNIST, CIFAR
from training import Results
from results import image_plot


from runcolor import models


experiment_name = "ar_conjoined"

train_steps = 50000
save_steps = 10000

suffix = None

data = MNIST()    # MNIST, CIFAR


tracker = Results.load(experiment_name)  # performance tracker

syntheses = ['reconstruct', 'sample', 'fix_latents', 'latent_activations']


for i in range(save_steps, train_steps+save_steps, save_steps):

    suffix = str(i)

    image_plot(tracker, models, data=data, suffix=suffix, syntheses=syntheses,
            n_rows=8, n_cols=8, n_pixels=0, spacing=0)
