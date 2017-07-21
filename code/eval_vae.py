from models import vae
from data import MNIST

from training import Results


tracker = Results.load('trial_vae')


# parameters
parms = {
    # basic parameters
    'n_z': 200,
    'height': 28,
    'width': 28,
    'n_channels': 1,

    # network parameters
    'n_units': 200,
    'n_feature_maps': 32,

    # autoregressive model parameters
    'n_pixelcnn_layers': 2,
    'conditional': True,
    'concat': True,

    # loss function parameters
    'anneal': -0.25,  # -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.002,
    'batch_size': 64,
    'n_conditional_pixels': 300,
    'test_sample_size': 100,
    'train_steps': 100,
    'test_steps': 10,
    'save_steps': 50
}

# data
mnist = MNIST()


# plot curves
curve_plot()

# synthesize data
image_plot()

    train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)





