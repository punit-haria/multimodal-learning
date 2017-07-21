from models import vae
from data import MNIST

from training import train, Results


experiment_name = 'vae_fc'


models = {
    #'VAE': vae.VAE,
    'VAE_AR': vae.VAE_AR
    #'VAE_CNN': vae.VAE_CNN,
    #'VAE_CNN_AR': vae.VAE_CNN_AR
}

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
    'n_pixelcnn_layers': 6,
    'conditional': True,
    'concat': True,

    # loss function parameters
    'anneal': -0.25,  # -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.002,
    'batch_size': 64,
    'n_conditional_pixels': 300,
    'test_sample_size': 1000,
    'train_steps': 10000,
    'test_steps': 50,
    'save_steps': 5000
}


mnist = MNIST()    # data

tracker = Results(experiment_name)  # performance tracker


for cond in [False, True]:
    parms['conditional'] = cond

    for name, model in models.items():

        name = name + '_conditioned_' + str(cond)

        train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)





