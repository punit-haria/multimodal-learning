from data import MNIST
from training import Results
from results import image_plot


from run_vae import models, parms, experiment_name


parms['n_conditional_pixels'] = 300
suffix = str(parms['train_steps'])


tracker = Results.load(experiment_name)  # performance tracker
names = tracker.get_runs()

mnist = MNIST()  # data



for name, model in models.items():

    image_plot()

    train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)






