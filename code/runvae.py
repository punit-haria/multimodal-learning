from models import vae
from data import MNIST

from training import train, Results


experiment_name = 'testing_new_code'


models = [
    vae.VAE
]
models = {x.__name__: x for x in models}


# parameters
parms = {
    # options
    "type": "cnn",              # fc, cnn
    "data": "mnist",            # mnist
    "autoregressive": False,
    "flow": False,

    # basic parameters
    'n_z': 32, #200,
    'height': 28,
    'width': 28,
    'n_channels': 1,

    # network parameters
    'n_units': 450,
    'n_feature_maps': 16,

    # normalizing flow parameters
    'flow_units': 32,
    'flow_layers': 4,

    # autoregressive model parameters
    'n_pixelcnn_layers': 2,

    # loss function parameters
    'anneal': 0,  # 0, -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.002,
    'batch_size': 64, # 256

    'n_conditional_pixels': 300,
    'test_sample_size': 1000,
    'train_steps': 5000,
    'test_steps': 50,
    'save_steps': 5000
}


if __name__ == "__main__":

    type = ["fc"]
    flows = [True]
    areg = [False]

    mnist = MNIST()    # data
    tracker = Results(experiment_name)  # performance tracker

    for t in type:
        for flow in flows:
            for ar in areg:

                for name, model in models.items():

                    parms['type'] = t
                    parms['autoregressive'] = ar
                    parms['flow'] = flow

                    name = experiment_name + "_" + name +'_' + t + '_flow_' + str(flow) + '_ar_' + str(ar)

                    train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)

