from models import vae2 as vae
from data import CIFAR

from training import train, Results


experiment_name = 'mixture'   # decoder, reconstruction, decode?


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

    # basic parameters
    'n_z': 100,  # 32, 49, 200
    'height': 32,
    'width': 32,
    'n_channels': 3,

    # network parameters
    'n_units': 500,
    'n_feature_maps': 64,

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
    'batch_size': 128,
    'n_conditional_pixels': 300,
    'test_sample_size': 500,
    'train_steps': 20000,
    'test_steps': 50,
    'save_steps': 20000
}


if __name__ == "__main__":

    # data, type, flow, flow_layers, flow_units, flow_type, autoregressive, n_ar_layers, anneal

    configs = [
        ["cnn", False, 4, 1024, "made", False, 6, 0]
    ]

    data = CIFAR()

    tracker = Results(experiment_name)  # performance tracker

    for c in configs:

        parms['type'] = c[0]

        parms['flow'] = c[1]
        parms['flow_layers'] = c[2]
        parms['flow_units'] = c[3]
        parms['flow_type'] = c[4]

        parms['autoregressive'] = c[5]
        parms['n_pixelcnn_layers'] = c[6]

        parms['anneal'] = c[7]

        for name, model in models.items():

            name =  experiment_name + "_cifar_" + c[0]

            if c[1]:
                name += "_flow_" + str(c[2]) + "_" + str(c[3]) + "_" + c[4]

            if c[5]:
                name += "_autoregressive_" + str(c[6])

            if c[7] < 0:
                name += "_anneal_" + str(c[7])

            train(name=name, model=model, parameters=parms, data=data, tracker=tracker)


    print("Finished :)", flush=True)