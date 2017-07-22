from data import MNIST
from training import Results
from results import image_plot


from run_vae import models, parms, experiment_name


suffix = str(parms['train_steps'])


tracker = Results.load(experiment_name)  # performance tracker
mnist = MNIST()  # data


image_plot(tracker, models, data=mnist, suffix=suffix, n_rows=8, n_cols=8, n_pixels=300,
           spacing=0, synthesis_type='reconstruct')

