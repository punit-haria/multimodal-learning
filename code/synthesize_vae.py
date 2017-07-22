from data import MNIST
from training import Results
from results import image_plot


from run_vae import models, parms, experiment_name


parms['n_conditional_pixels'] = 300
suffix = str(parms['train_steps'])


tracker = Results.load(experiment_name)  # performance tracker
mnist = MNIST()  # data


for name, model in models.items():

    image_plot(tracker, models, parms, data=mnist, suffix=suffix, n_rows=8, n_cols=8,
               spacing=0, synthesis_type='reconstruct')


