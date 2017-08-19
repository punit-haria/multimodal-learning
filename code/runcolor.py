from models import vae
from data import CIFAR

from training import train, Results


experiment_name = 'global_lossy_cifar'

models = [
    vae.VAE
]
models = {x.__name__: x for x in models}


# parameters
parms = {
    # options
    'type': "cnn",              # fc, cnn
    'data': "cifar",            # cifar
    'autoregressive': False,
    'flow': False,
    'output': 'continuous',     # discrete, continuous

    # basic parameters
    'n_z': 200,  # 200
    'height': 32,
    'width': 32,
    'n_channels': 3,
    'n_mixtures': 5,  # 5

    # network parameters
    'n_units': 450,
    'n_feature_maps': 64,  # 64

    # normalizing flow parameters
    'flow_units': 320,
    'flow_layers': 2,
    'flow_type': "cnn",  # cnn, made

    # autoregressive model parameters
    'n_pixelcnn_layers': 2,

    # loss function parameters
    'anneal': -0.5,  # 0, -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.001,  # 0.001
    'batch_size': 256,
    'n_conditional_pixels': 0,
    'test_sample_size': 500,
    'train_steps': 5000,
    'test_steps': 50,
    'save_steps': 5000
}


if __name__ == "__main__":

    # data, type, flow, flow_layers, flow_units, flow_type, autoregressive, n_ar_layers, anneal, n_z, n_mix, lr

    configs = [
        ["cnn", "continuous", False, 4, 1024, "made", True, 1, -0.5, 200, 5, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", True, 2, -0.5, 200, 5, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", True, 2, -0.25, 200, 5, 0.001]
    ]
    data = CIFAR()

    tracker = Results(experiment_name)  # performance tracker

    for c in configs:

        parms['type'] = c[0]
        parms['output'] = c[1]

        parms['flow'] = c[2]
        parms['flow_layers'] = c[3]
        parms['flow_units'] = c[4]
        parms['flow_type'] = c[5]

        parms['autoregressive'] = c[6]
        parms['n_pixelcnn_layers'] = c[7]

        parms['anneal'] = c[8]

        parms['n_z'] = c[9]
        parms['n_mixtures'] = c[10]
        parms['learning_rate'] = c[11]

        for name, model in models.items():

            name =  experiment_name + '_nz_' +  str(parms['n_z']) + '_nmix_' + str(parms['n_mixtures']) \
                    + '_lr_' + str(parms['learning_rate'])

            if parms['flow']:
                name += "_flow_" + str(parms['flow_layers']) + "_" + str(parms['flow_units']) + "_" + parms['flow_type']

            if parms['autoregressive']:
                name += "_ar_" + str(parms['n_pixelcnn_layers'])

            if parms['anneal'] < 0:
                name += "_anneal_" + str(parms['anneal'])

            train(name=name, model=model, parameters=parms, data=data, tracker=tracker)


    print("Finished :)", flush=True)