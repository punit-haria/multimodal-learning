from models import vae
from data import MNIST

from training import train, Results


experiment_name = 'vae_cnn_vs_fc'


models = [
    vae.VAE,
    vae.VAE_CNN
]
models = {x.__name__: x for x in models}


# parameters
parms = {
    # basic parameters
    'n_z': 200,
    'height': 28,
    'width': 28,
    'n_channels': 1,

    # network parameters
    'n_units': 450,
    'n_feature_maps': 16,

    # autoregressive model parameters
    'n_pixelcnn_layers': 6,
    'concat': False,

    # loss function parameters
    'anneal': -0.25,  # -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.002,
    'batch_size': 256,

    'n_conditional_pixels': 300,
    'test_sample_size': 1000,
    'train_steps': 5000,
    'test_steps': 50,
    'save_steps': 5000
}


if __name__ == "__main__":


    mnist = MNIST()    # data
    tracker = Results(experiment_name)  # performance tracker


    for name, model in models.items():

        if name == "VAE_CNN":

            for alpha in [0, -0.125, -0.25]:

                name = name + '_anneal_' + str(alpha)

                parms['anneal'] = alpha
                train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)

        else:

            train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)



