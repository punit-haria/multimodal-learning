from data import MNIST, CIFAR
from training import Results
from results import image_plot


from runcolor import models


experiment_name = "global_lossy_cifar"

train_steps = 150000
save_steps = 30000

suffix = None

data = CIFAR()    # MNIST, CIFAR


tracker = Results.load(experiment_name)  # performance tracker

syntheses = ['reconstruct', 'sample', 'fix_latents', 'latent_activations']


for i in range(save_steps, train_steps+save_steps, save_steps):

    suffix = str(i)

    image_plot(tracker, models, data=data, suffix=suffix, syntheses=syntheses,
            n_rows=8, n_cols=8, n_pixels=0, spacing=0, count=3)
