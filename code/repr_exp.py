from data import JointStratifiedMNIST, ColouredStratifiedMNIST
from training import Results
from runjoint import models
import numpy as np

from sklearn.neural_network import MLPClassifier


experiment_name = "discrete_colored_final"
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

    x = np.reshape(x, newshape=[-1, 784*3])

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

    print("Initializing model...")
    mod = _model(arguments=parms, name=name, tracker=tracker, init_minibatches=xs)

    print("Loading model state...")
    mod.load_state(suffix=suffix)


    mean = True

    print("Converting training data..")
    z1 = convert(mod, data.x1, left=True, bs=None, mean=mean)
    z2 = convert(mod, data.x2, left=False, bs=None, mean=mean)
    ytr = data.y1

    print("Converting test data..")
    z1_test = convert(mod, data.M1_test, left=True, bs=None, mean=mean)
    z2_test = convert(mod, data.M2_test, left=False, bs=None, mean=mean)
    yte = data.yte

    # Representation tests:

    print("Training with z1...", flush=True)
    mlp = MLPClassifier()
    mlp.fit(z1, ytr)
    print("Same-side score: ", mlp.score(z1_test, yte))             # 0.9663
    print("Cross score: ", mlp.score(z2_test, yte))                 # 0.7048

    print("Training with z2...", flush=True)
    mlp = MLPClassifier()
    mlp.fit(z2, ytr)
    print("Same-side score: ", mlp.score(z2_test, yte))             # 0.0959
    print("Cross score: ", mlp.score(z1_test, yte))                 # 0.1184


    # Raw tests:

    x1 = np.reshape(data.x1, newshape=[-1, 784 * 3])
    x2 = np.reshape(data.x2, newshape=[-1, 784 * 3])

    x1_test = np.reshape(data.M1_test, newshape=[-1, 784 * 3])
    x2_test = np.reshape(data.M2_test, newshape=[-1, 784 * 3])

    print("Training with x1...", flush=True)
    mlp = MLPClassifier()
    mlp.fit(x1, ytr)
    print("Same-side score: ", mlp.score(x1_test, yte))             # 0.9531
    print("Cross score: ", mlp.score(x2_test, yte))                 # 0.5165

    print("Training with x2...", flush=True)
    mlp = MLPClassifier()
    mlp.fit(x2, ytr)
    print("Same-side score: ", mlp.score(x2_test, yte))             # 0.0878
    print("Cross score: ", mlp.score(x1_test, yte))                 # 0.0855







