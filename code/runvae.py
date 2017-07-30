from models import vae
from data import MNIST, ColouredMNIST

from training import train, Results


experiment_name = 'exp'


models = [
    vae.VAE
]
models = {x.__name__: x for x in models}


# parameters
parms = {
    # options
    'type': "cnn",              # fc, cnn
    'data': "mnist",            # mnist, color
    'autoregressive': False,
    'flow': False,

    # basic parameters
    'n_z': 49,  # 32, 49, 200
    'height': 28,
    'width': 28,
    'n_channels': 1,

    # network parameters
    'n_units': 450,
    'n_feature_maps': 32,

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
    'batch_size': 128,
    'n_conditional_pixels': 300,
    'test_sample_size': 1000,
    'train_steps': 10000,
    'test_steps': 50,
    'save_steps': 500
}


if __name__ == "__main__":

    # data, type, flow, flow_layers, flow_units, flow_type, autoregressive, n_ar_layers, anneal

    configs = [
        ["mnist", "cnn", False, 1, 320, "made", True, 6, -0.25]
    ]

    if configs[0][0] == "color":
        data = ColouredMNIST(50000)
    else:
        data = MNIST()

    tracker = Results(experiment_name)  # performance tracker

    for c in configs:

        parms['data'] = c[0]
        if c[0] == "color":
            parms['n_channels'] = 3

        parms['type'] = c[1]

        parms['flow'] = c[2]
        parms['flow_layers'] = c[3]
        parms['flow_units'] = c[4]
        parms['flow_type'] = c[5]

        parms['autoregressive'] = c[6]
        parms['n_pixelcnn_layers'] = c[7]

        parms['anneal'] = c[8]

        for name, model in models.items():

            name = experiment_name + "_" + c[0] + "_" + c[1]

            if c[2]:
                name += "_flow_" + str(c[3]) + "_" + str(c[4]) + "_" + c[5]

            if c[6]:
                name += "_autoregressive_" + str(c[7])

            if c[8] < 0:
                name += "_anneal_" + str(c[8])

            train(name=name, model=model, parameters=parms, data=data, tracker=tracker)


    print("Finished :)", flush=True)