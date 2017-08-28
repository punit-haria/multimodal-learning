from data import JointStratifiedMNIST, ColouredStratifiedMNIST
from training import Results
from runjoint import models
import numpy as np

from sklearn.neural_network import MLPClassifier


experiment_name = "final_discrete_color"
runs = ['discrete_colored_final_cnn_small_nz_64_lr_0.001_fmaps_32_units_128_obj_joint_jointanneal_0.3']



tracker = Results.load(experiment_name)

data = ColouredStratifiedMNIST(2000)


def sample(data, n_samples, dtype='test'):

    if dtype == 'test':
        x1, x2 = data.sample_stratified(n_paired_samples=n_samples, dtype='test')
        return x1, x2

    else:
        paired, unpaired = n_samples
        x1, x2, x1p, x2p = data.sample_stratified(n_paired_samples=paired, n_unpaired_samples=unpaired,
                                                  dtype='train')
        return x1, x2, x1p, x2p


def convert(model, x, left, bs, mean):

    n = len(data.x1)
    if bs is None:
        bs =  n

    zs = []

    for b in range(0, n, bs):
        x = x[b:b+bs]
        if left:
            z = model.encode((x, None), mean=mean)
        else:
            z = model.encode((None, x), mean=mean)
        zs.append(z)

    z = np.concatenate(zs, axis=0)
    return z



for name in tracker.get_runs():

    if name not in runs:
        continue

    trial = tracker.get(name)
    _model = models[trial.model_name]
    parms = trial.parameters

    suffix = str(parms['train_steps'])

    prd = parms['n_paired_samples']
    uprd = parms['n_unpaired_samples']
    xs = sample(data, n_samples=(prd, uprd), dtype='train')

    mod = _model(arguments=parms, name=name, tracker=tracker, init_minibatches=xs)

    mod.load_state(suffix=suffix)

    # training data
    print("Converting training data..")
    z = convert(mod, data.x1, left=True, bs=None, mean=True)  # 1000
    y = data.y1

    mlp = MLPClassifier()
    mlp.fit(z, y)


    # cross test data
    print("Converting test data..")
    zte = convert(mod, data.M2_test, left=False, bs=None, mean=True)
    yte = data.yte

    cross_score = mlp.score(zte, yte)


    # same test data
    print("Converting test data..")
    zte = convert(mod, data.M1_test, left=True, bs=None, mean=True)
    yte = data.yte

    same_score = mlp.score(zte, yte)


    print("Cross score: ", cross_score)

    print("Same score: ", same_score)









