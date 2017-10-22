from models import multimodalvae
from data import MSCOCO

from training import train_joint, Results


experiment_name = 'coco_cnn'


models = [
    multimodalvae.MultiModalVAE
]
models = {x.__name__: x for x in models}


# parameters
parms = {
    # basic parameters
    'objective': 'joint',    # joint, translate
    'n_z': 128,

    # network parameters
    'n_units_image': 256,
    'n_units_enc_capt': 512,
    'n_feature_maps_image': 64,
    'n_feature_maps_capt': 512,
    'embed_size': 1024,
    'gru_layers': 1,
    'softmax_samples': 6000,

    # loss function parameters
    'anneal': 0,  # 0, -0.0625, -0.125, -0.25   (0 is no anneal)
    'joint_anneal': 0.3,   # 0.1, 0.3, 0.5      (1 is no anneal)

    # train/test parameters
    'learning_rate': 0.001,
    'n_unpaired_samples': 128,
    'n_paired_samples': 128,
    'n_paired': 60000,

    'test_sample_size': 128,
    'train_steps': 30000,  # 350000
    'test_steps': 50,
    'save_steps': 3000
}


if __name__ == "__main__":

    # objective, n_z, n_units_image, n_units_enc_capt, n_feature_maps_image, n_feature_maps_capt,
    # embed_size, gru_layers, softmax_samples, anneal, joint_anneal, learning_rate

    configs = [
        ['joint', 128, 256, 512, 64, 512, 1024, 1, 6000, 0, 0.3, 0.001]
    ]

    data = MSCOCO(parms['n_paired'])

    parms['max_seq_len'] = data.get_max_seq_len()    # max sequence length
    parms['vocab_size'] = data.get_vocab_size()     # vocabulary size

    print("Max seq. length: ", parms['max_seq_len'], flush=True)
    print("Vocabulary size: ", parms['vocab_size'], flush=True)

    tracker = Results(experiment_name)  # performance tracker

    for c in configs:
        parms['objective'] = c[0]
        parms['n_z'] = c[1]

        parms['n_units_image'] = c[2]
        parms['n_units_enc_capt'] = c[3]
        parms['n_feature_maps_image'] = c[4]
        parms['n_feature_maps_capt'] = c[5]

        parms['embed_size'] = c[4]
        parms['gru_layers'] = c[5]

        parms['softmax_samples'] = c[6]

        parms['anneal'] = c[7]
        parms['joint_anneal'] = c[8]

        parms['learning_rate'] = c[9]


        for name, model in models.items():

            name =  experiment_name + '_obj_' + parms['objective'] \
                    + '_nz_' +  str(parms['n_z']) \
                    + '_lr_' + str(parms['learning_rate']) \
                    + '_units_' + str(parms['n_units_image']) + '_' + str(parms['n_units_enc_capt']) \
                    + '_fmaps_' + str(parms['n_feature_maps_image']) + '_' + str(parms['n_feature_maps_capt']) \
                    + '_embed_' + str(parms['embed_size']) \
                    + '_gru_' + str(parms['gru_layers']) \
                    + '_sftsamples_' + str(parms['softmax_samples'])

            if parms['anneal'] < 0:
                name += "_anneal_" + str(parms['anneal'])

            if parms['joint_anneal'] < 1:
                name += "_jointanneal_" + str(parms['joint_anneal'])

            train_joint(name=name, model=model, parameters=parms, data=data, tracker=tracker)


    print("Finished :)", flush=True)

