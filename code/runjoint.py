from models import jointvae
from data import JointStratifiedMNIST, ColouredStratifiedMNIST

from training import train_joint, Results


experiment_name = 'joint'


models = [
    jointvae.JointVAE
]
models = {x.__name__: x for x in models}


# parameters
parms = {
    # options
    'type': "fc",              # fc, cnn
    'data': "mnist",            # mnist, ???
    'autoregressive': False,
    'flow': False,
    'output': 'continuous',     # discrete, continuous
    'objective': 'joint',    # joint, translate

    # basic parameters
    'n_z': 20,  # 32, 49, 200
    'height': 28,
    'width': 28,
    'n_channels': 3,
    'n_mixtures': 5,

    # network parameters
    'n_units': 200,
    'n_feature_maps': 32,  # 32

    # normalizing flow parameters
    'flow_units': 320,
    'flow_layers': 2,
    'flow_type': "made",  # cnn, made

    # autoregressive model parameters
    'n_pixelcnn_layers': 3,

    # loss function parameters
    'anneal': 0,  # 0, -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.001,
    'n_unpaired_samples': 96,  # 96
    'n_paired_samples': 32,    # 32

    'n_paired': 1000,
    'n_conditional_pixels': 0,
    'test_sample_size': 128,
    'train_steps': 10000,
    'test_steps': 50,
    'save_steps': 10000
}


if __name__ == "__main__":

    # data, type, flow, flow_layers, flow_units, flow_type, autoregressive, n_ar_layers, anneal

    configs = [
        ["cnn", False, 4, 1024, "made", False, 6, 0]
    ]

    data = ColouredStratifiedMNIST(parms['n_paired'])
    #data = JointStratifiedMNIST(parms['n_paired'])

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

            name =  experiment_name + "_" + parms['data'] + "_" + parms['type']

            if parms['flow']:
                name += "_flow_" + str(parms['flow_layers']) + "_" + str(parms['flow_units']) + "_" + parms['flow_type']

            if parms['autoregressive']:
                name += "_autoregressive_" + str(parms['n_pixelcnn_layers'])

            if c[7] < 0:
                name += "_anneal_" + str(parms['anneal'])

            train_joint(name=name, model=model, parameters=parms, data=data, tracker=tracker)


    print("Finished :)", flush=True)