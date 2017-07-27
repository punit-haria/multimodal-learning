from models import vae
from data import MNIST

from training import train, Results


experiment_name = '13_trials'


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

    # basic parameters
    'n_z': 49,  # 32, 200
    'height': 28,
    'width': 28,
    'n_channels': 1,

    # network parameters
    'n_units': 450,
    'n_feature_maps': 16,

    # normalizing flow parameters
    'flow_units': 512, #320,
    'flow_layers': 1,
    'flow_type': "cnn",  # cnn, fc

    # autoregressive model parameters
    'n_pixelcnn_layers': 3,

    # loss function parameters
    'anneal': 0,  # 0, -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.002,
    'batch_size': 128,
    'n_conditional_pixels': 300,
    'test_sample_size': 1000,
    'train_steps': 30000,
    'test_steps': 50,
    'save_steps': 30000
}


if __name__ == "__main__":

    # type, flow, autoregressive, flow_layers, flow_units, flow_type
    configs = [
        ["cnn", False, False, 1, 512, "fc"],
        ["cnn", False, True, 1, 512, "fc"],

        ["fc", True, False, 2, 512, "fc"],
        ["fc", True, False, 4, 512, "fc"],
        ["fc", True, False, 4, 1536, "fc"],
        ["fc", True, False, 8, 512, "fc"],

        ["cnn", True, False, 2, 512, "fc"],
        ["cnn", True, False, 4, 1536, "fc"],

        ["cnn", True, True, 2, 512, "fc"],
        ["cnn", True, True, 4, 1536, "fc"],

        ["cnn", True, False, 2, 512, "cnn"],
        ["cnn", True, False, 4, 512, "cnn"],
        ["cnn", True, True, 2, 512, "cnn"],
    ]


    mnist = MNIST()    # data
    tracker = Results(experiment_name)  # performance tracker

    for c in configs:

        parms['type'] = c[0]
        parms['flow'] = c[1]
        parms['autoregressive'] = c[2]
        parms['flow_layers'] = c[3]
        parms['flow_units'] = c[4]
        parms['flow_type'] = c[5]

        for name, model in models.items():

            #name = experiment_name + "_" + name

            name = experiment_name + "_" + c[0]
            if c[2]:
                name += "_autoregressive"
            if c[1]:
                name += "_flow_" + str(c[3]) + "_" + str(c[4]) + "_" + c[5]

            train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)

