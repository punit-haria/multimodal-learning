from data import MNIST
from training import Results
from run_vae import parms


tracker = Results.load('trial_vae')

parms['n_conditional_pixels'] = 300

suffix = str(parms['train_steps'])


mnist = MNIST()  # data



# plot curves
curve_plot()

# synthesize data
image_plot()

    train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)





