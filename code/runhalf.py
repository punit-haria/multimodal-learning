from models import jointvae
from data import JointStratifiedMNIST

from training import train_joint, Results


experiment_name = 'halved_mnist'


models = [
    jointvae.JointVAE
]
models = {x.__name__: x for x in models}


# parameters
parms = {
    # options
    'type': "cnn",              # fc, cnn
    'data': "halved_mnist",      # halved_mnist, mnist
    'autoregressive': False,
    'flow': False,
    'output': 'discrete',     # discrete, continuous
    'objective': 'joint',    # joint, translate
    'joint_type': 'small',   # constrain, small, large

    # basic parameters
    'n_z': 64,  # 32, 49, 200
    'height': 14,
    'width': 28,
    'n_channels': 1,
    'n_mixtures': 5,

    # network parameters
    'n_units': 96,
    'n_feature_maps': 16,

    # normalizing flow parameters
    'flow_units': 320,
    'flow_layers': 2,
    'flow_type': "made",  # cnn, made

    # autoregressive model parameters
    'n_pixelcnn_layers': 2,

    # loss function parameters
    'anneal': -0.25,  # 0, -0.0625, -0.125, -0.25

    # train/test parameters
    'learning_rate': 0.001,
    'n_unpaired_samples': 256,
    'n_paired_samples': 64,

    'n_paired': 1000,
    'n_conditional_pixels': 0,
    'test_sample_size': 1000,
    'train_steps': 150000,
    'test_steps': 50,
    'save_steps': 30000
}


if __name__ == "__main__":

    # data, type, flow, flow_layers, flow_units, flow_type, autoregressive, n_ar_layers, anneal,
    # joint_type, n_z, n_mix, lr, n_units, n_fmaps, objective,  temp_weight

    configs = [                         # maybe reduce learning rate to 0.001????
        ["cnn", "discrete", False, 4, 1024, "made", False, 2, 0, 'small', 64, 5, 0.002, 128, 16, 'joint', 0.5],
        ["cnn", "discrete", False, 4, 1024, "made", False, 2, 0, 'small', 64, 5, 0.002, 128, 16, 'joint', 0.3]
        ["cnn", "discrete", False, 4, 1024, "made", False, 2, 0, 'small', 64, 5, 0.002, 128, 16, 'joint', 0.8]
    ]

    data = JointStratifiedMNIST(parms['n_paired'])

    tracker = Results(experiment_name)  # performance tracker

    for c in configs:
        parms['tw'] = c[16]

        parms['type'] = c[0]
        parms['output'] = c[1]

        parms['flow'] = c[2]
        parms['flow_layers'] = c[3]
        parms['flow_units'] = c[4]
        parms['flow_type'] = c[5]

        parms['autoregressive'] = c[6]
        parms['n_pixelcnn_layers'] = c[7]

        parms['anneal'] = c[8]

        parms['joint_type'] = c[9]

        parms['n_z'] = c[10]
        parms['n_mixtures'] = c[11]
        parms['learning_rate'] = c[12]
        parms['n_units'] = c[13]
        parms['n_feature_maps'] = c[14]

        parms['objective'] = c[15]


        for name, model in models.items():

            name =  experiment_name + '_' + parms['type'] + '_' + parms['joint_type'] \
                    + '_nz_' +  str(parms['n_z']) \
                    + '_lr_' + str(parms['learning_rate']) \
                    + '_fmaps_' + str(parms['n_feature_maps']) \
                    + '_units_' + str(parms['n_units']) \
                    + '_obj_' + parms['objective']

            if parms['output'] == "continuous":
                name += '_nmix_' + str(parms['n_mixtures'])


            if parms['flow']:
                name += "_flow_" + str(parms['flow_layers']) + "_" + str(parms['flow_units']) + "_" + parms['flow_type']

            if parms['autoregressive']:
                name += "_ar_" + str(parms['n_pixelcnn_layers'])

            if parms['anneal'] < 0:
                name += "_anneal_" + str(parms['anneal'])

            train_joint(name=name, model=model, parameters=parms, data=data, tracker=tracker)


    print("Finished :)", flush=True)