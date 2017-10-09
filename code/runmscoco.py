from models import multimodalvae
from data import MSCOCO

from training import train_joint, Results


experiment_name = 'coco'


models = [
    multimodalvae.MultiModalVAE
]
models = {x.__name__: x for x in models}


# parameters
parms = {
    # basic parameters
    'objective': 'joint',    # joint, translate
    'n_z': 50,

    # network parameters
    'n_feature_maps': 32,
    'n_units': 75,
    'embed_size': 80,
    'gru_layers': 1,

    # loss function parameters
    'anneal': 0,  # 0, -0.0625, -0.125, -0.25   (0 is no anneal)
    'joint_anneal': 0.3,   # 0.1, 0.3, 0.5      (1 is no anneal)

    # train/test parameters
    'learning_rate': 0.001,
    'n_unpaired_samples': 64,
    'n_paired_samples': 64,

    'n_paired': 40000,
    'test_sample_size': 128,
    'train_steps': 10000,  # 350000
    'test_steps': 50,
    'save_steps': 2000
}


if __name__ == "__main__":

    # objective, n_z, n_feature_maps, n_units, embed_size, gru_layers, anneal, joint_anneal, learning_rate

    configs = [
        ['joint', 50, 32, 75, 80, 1, 0, 0.3, 0.001]
    ]

    data = MSCOCO(parms['n_paired'])

    parms['max_seq_len'] = data.get_max_seq_len()    # max sequence length
    parms['vocab_size'] = data.get_vocab_size()     # vocabulary size


    tracker = Results(experiment_name)  # performance tracker

    for c in configs:
        parms['objective'] = c[0]
        parms['n_z'] = c[1]

        parms['n_feature_maps'] = c[2]
        parms['n_units'] = c[3]
        parms['embed_size'] = c[4]
        parms['gru_layers'] = c[5]

        parms['anneal'] = c[6]
        parms['joint_anneal'] = c[7]

        parms['learning_rate'] = c[8]


        for name, model in models.items():

            name =  experiment_name + '_obj_' + parms['objective'] \
                    + '_nz_' +  str(parms['n_z']) \
                    + '_lr_' + str(parms['learning_rate']) \
                    + '_fmaps_' + str(parms['n_feature_maps']) \
                    + '_units_' + str(parms['n_units']) \
                    + '_embed_' + str(parms['embed_size']) \
                    + '_gru_' + str(parms['gru_layers'])

            if parms['anneal'] < 0:
                name += "_anneal_" + str(parms['anneal'])

            if parms['joint_anneal'] < 1:
                name += "_jointanneal_" + str(parms['joint_anneal'])

            train_joint(name=name, model=model, parameters=parms, data=data, tracker=tracker)


    print("Finished :)", flush=True)

