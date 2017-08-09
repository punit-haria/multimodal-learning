from models import vae
from data import MNIST, ColouredMNIST

from training import train, Results


experiment_name = 'ar_test'


models = [
    vae.VAE
]
models = {x.__name__: x for x in models}


# parameters
parms = {
    # options
    'type': "cnn",              # fc, cnn
    'data': "mnist",            # mnist
    'autoregressive': False,
    'flow': False,
    'output': 'discrete',     # discrete, continuous

    # basic parameters
    'n_z': 49,  # 32, 49, 200
    'height': 28,
    'width': 28,
    'n_channels': 1,
    'n_mixtures': 5,

    # network parameters
    'n_units': 450,
    'n_feature_maps': 16,

    # normalizing flow parameters
    'flow_units': 320,
    'flow_layers': 2,
    'flow_type': "cnn",  # cnn, made

    # autoregressive model parameters
    'n_pixelcnn_layers': 3,

    # loss function parameters
    'anneal': 0,  # 0, -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.002,
    'batch_size': 256,
    'n_conditional_pixels': 0,
    'test_sample_size': 1000,
    'train_steps': 10000,
    'test_steps': 50,
    'save_steps': 10000
}


if __name__ == "__main__":

    # data, type, flow, flow_layers, flow_units, flow_type, autoregressive, n_ar_layers, anneal

    configs = [
        ["cnn", "discrete", False, 4, 1024, "made", False, 3, 0],
        ["cnn", "discrete", False, 4, 1024, "made", True, 3, 0],
        ["cnn", "discrete", False, 4, 1024, "made", True, 3, -0.125],
        ["cnn", "discrete", False, 4, 1024, "made", True, 3, -0.25],
        ["cnn", "discrete", False, 4, 1024, "made", True, 3, -0.5]
    ]

    data = MNIST()

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

        for name, model in models.items():

            name =  experiment_name + "_cifar_" + parms['type'] + "_" + parms['output']

            if parms['flow']:
                name += "_flow_" + str(parms['flow_layers']) + "_" + str(parms['flow_units']) + "_" + parms['flow_type']

            if parms['autoregressive']:
                name += "_autoregressive_" + str(parms['n_pixelcnn_layers'])

            if parms['anneal'] < 0:
                name += "_anneal_" + str(parms['anneal'])

            train(name=name, model=model, parameters=parms, data=data, tracker=tracker)


    print("Finished :)", flush=True)
