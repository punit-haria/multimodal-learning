from models.vae import VAE
from data import MNIST

from training import train, Results


tracker = Results('experiment_vae_conditional_pixelcnn')

models = {
    'VAE': VAE
}

# parameters
parms = {
    # basic parameters
    'n_z': 200,
    'height': 28,
    'width': 28,
    'n_channels': 1,

    # fully connected layer parameters
    'n_units': 200,

    # autoregressive model parameters
    'n_pixelcnn_layers': 2,
    'conditional': True,
    'concat': True,

    # convolution network parameters
    'n_feature_maps': 32,

    # loss function parameters
    'anneal': -0.25,  # -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.002,
    'batch_size': 64,
    'test_sample_size': 1000,
    'n_conditional_pixels': 300,
    'train_steps': 10000,
    'test_steps': 50,
    'save_steps': 5000
}


# data
mnist = MNIST()


# train models
for name, model in models.items():
    train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)





