from models import vae
from data import CIFAR

from training import train, Results


experiment_name = 'cifar'

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
    'n_z': 100,  # 32, 49, 200
    'height': 32,
    'width': 32,
    'n_channels': 3,
    'n_mixtures': 5,

    # network parameters
    'n_units': 450,
    'n_feature_maps': 64,  # 64

    # normalizing flow parameters
    'flow_units': 320,
    'flow_layers': 2,
    'flow_type': "cnn",  # cnn, made

    # autoregressive model parameters
    'n_pixelcnn_layers': 3,

    # loss function parameters
    'anneal': 0,  # 0, -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.001,
    'batch_size': 256,
    'n_conditional_pixels': 0,
    'test_sample_size': 500,
    'train_steps': 1000,
    'test_steps': 50,
    'save_steps': 10000
}


if __name__ == "__main__":

    # data, type, flow, flow_layers, flow_units, flow_type, autoregressive, n_ar_layers, anneal, n_z, n_mix, lr

    configs = [
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 50, 3, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 100, 3, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 200, 3, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 500, 3, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 50, 5, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 100, 5, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 200, 5, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 500, 5, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 50, 10, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 100, 10, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 200, 10, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 500, 10, 0.001],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 50, 3, 0.0005],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 100, 3, 0.0005],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 200, 3, 0.0005],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 500, 3, 0.0005],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 50, 5, 0.0005],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 100, 5, 0.0005],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 200, 5, 0.0005],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 500, 5, 0.0005],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 50, 10, 0.0005],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 100, 10, 0.0005],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 200, 10, 0.0005],
        ["cnn", "continuous", False, 4, 1024, "made", False, 6, 0, 500, 10, 0.0005]
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