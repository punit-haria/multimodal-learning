import tensorflow as tf

from models.vae import VAE, VAECNN
from data import MNIST

from training import train, Results


# store experimental results
tracker = Results('experiment_vae_conditional_pixelcnn')

models = {
    'VAE_conditional_pixelCNN': VAE
    #'VAE_CNN': VAECNN
}

# parameters
parms = {
    'n_z': 200,
    'learning_rate': 0.002,
    'batch_size': 64,

    'n_channels': 1,
    'n_pixelcnn_layers': 2,
    'concat': True,
    'n_pixels': 300,

    'train_steps': 10000,
    'test_steps': 50,
    'save_steps': 5000,
    'test_sample_size': 1000
}


# data
mnist = MNIST()


# train models
for name, model in models.items():
    train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)





