from models import vae
from data import MNIST

from training import train, Results


experiment_name = 'vae_fc_ar'


models = [
    vae.VAE_AR
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
    'n_units': 200,
    'n_feature_maps': 32,

    # autoregressive model parameters
    'n_pixelcnn_layers': 6,
    'conditional': True,
    'concat': False,

    # loss function parameters
    'anneal': -0.25,  # -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.002,
    'batch_size': 128,
    'n_conditional_pixels': 300,
    'test_sample_size': 1000,
    'train_steps': 10000,
    'test_steps': 50,
    'save_steps': 5000
}


if __name__ == "__main__":

    mnist = MNIST()    # data
    tracker = Results(experiment_name)  # performance tracker


    # train models
    for cond in [(False, False), (True, False), (True, True)]:
        parms['conditional'] = cond[0]
        parms['concat'] = cond[1]

        for name, model in models.items():

            if cond[0]:
                name = name + '_conditioned_' + str(cond[0]) + '_concat_' + str(cond[1])
            else:
                name = name + '_conditioned_' + str(cond[0])

            train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)





