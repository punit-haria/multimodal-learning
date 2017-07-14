import numpy as np
import tensorflow as tf

from models.joint_vae import VAE, VAETranslate
from data import JointStratifiedMNIST as MNIST

from results import Results


# parameters
parms = {
    'n_z': 50,
    'n_x1': 392,
    'n_x2': 392,
    'n_units': 200,
    'learning_rate': 0.002,

    'n_paired': 1000,
    'image_dim': (14, 28, 1),

    'batch_size': 250,
    'n_unpaired_samples': 200,
    'n_paired_samples': 50,

    'train_steps': 10000,
    'plot_steps': 2500
}

models = {
    'vae_joint': VAE,
    'vae_translate': VAETranslate
}

# data
mnist = MNIST(parms['n_paired'])


# store experimental results
results = Results('experiment_mnist')

for name, model in models.items():

    # load model
    print("Training Model: ", name, flush=True)
    vae = model(arguments=parms, name=name)

    # store next experimental run
    results.create_run(name)

    # train model
    for i in range(parms['train_steps'] + 1):

        # random minibatch
        x1, x2, x1p, x2p = mnist.sample(parms['n_paired_samples'], parms['n_unpaired_samples'],
                                        dtype='train', binarize=True)

        # training step
        bound = vae.train(x1, x2, x1p, x2p)

        # save results
        results.add(i, bound, "training_curve")

        if i % 25 == 0:
            print("At iteration ", i)

            # test minibatch
            x1, x2 = mnist.sample(1000, dtype='test', binarize=False)

            # test model
            bound = vae.test(x1, x2)

            # save results
            results.add(i, bound, "test_lower_bound")

            # plot reconstructions
            if i % parms['plot_steps'] == 0:
                n_examples = 100

                x1b = x1[0:n_examples]
                x2b = x2[0:n_examples]
                x12b = np.concatenate((x1b, x2b), axis=1)

                rx1_1, rx2_1 = vae.reconstruct_from_x1(x1b)
                rx1_2, rx2_2 = vae.reconstruct_from_x2(x2b)
                rx1p, rx2p = vae.reconstruct(x1b, x2b)

                # save reconstructions
                results.add(i, (x1b, rx1_1), "x1_1")
                results.add(i, (x1b, rx2_1), "x2_1")
                results.add(i, (x2b, rx1_2), "x1_2")
                results.add(i, (x2b, rx2_2), "x2_2")
                results.add(i, (x1b, x2b, rx1p, rx2p), "x12p")

    # save final model
    vae.save_state()

    # reset tensorflow session and graph
    vae.sess.close()
    tf.reset_default_graph()

# save experimental results
Results.save(results)