import numpy as np
import tensorflow as tf

from models.joint_vae import VAE, VAETranslate
from data import ColouredStratifiedMNIST as MNIST

from results import Results


print("Starting experiment...", flush=True)

### PARAMETERS ###

# parameters
parms = {
    'n_z': 50,
    'n_x1': 784 * 3,
    'n_x2': 784 * 3,
    'n_units': 500,
    'learning_rate': 0.001,

    'n_paired': 1000,
    'image_dim': (28, 28, 3),

    'n_unpaired_samples': 250,
    'n_paired_samples': 50,

    'train_steps': 5000,
    'plot_steps': 500
}


models = {
    'vae_joint': VAE,
    'vae_translate': VAETranslate
}

# data
print("Reading data...", flush=True)
mnist = MNIST(parms['n_paired'])


# store experimental results
results = Results('experiment_mnist')


for name, model in models.items():
    name = 'mnist_colored_'+name

    print("Loading model...", flush=True)
    vae = model(arguments=parms, name=name)

    # store next experimental run
    results.create_run(name)


    # train model
    print("Training...", flush=True)
    for i in range(parms['train_steps'] + 1):
        
        # random minibatch
        x1, x2, x1p, x2p = mnist.sample_stratified(n_paired_samples=parms['n_paired_samples'],
                                                   n_unpaired_samples=parms['n_unpaired_samples'], dtype='train')

        # training step
        bound = vae.train(x1, x2, x1p, x2p)

        # save results
        results.add(i, bound, "training_curve")

        if i % 50 == 0:
            print("At iteration ", i, flush=True)

            # test minibatch
            x1, x2 = mnist.sample_stratified(n_paired_samples=1000, dtype='test')

            # test model
            bound = vae.test(x1, x2)

            # save results
            results.add(i, bound, "test_lower_bound")

            # plot reconstructions 
            if i % parms['plot_steps'] == 0:
                n_examples = 18

                x1b = x1[0:n_examples]
                x2b = x2[0:n_examples]

                x12b = np.concatenate((x1b, x2b), axis=1)

                rx1_1, rx2_1 = vae.reconstruct_from_x1(x1b)
                rx1_2, rx2_2 = vae.reconstruct_from_x2(x2b)

                # save reconstructions
                results.add(i, (x1b, rx2_1), "x2_1")
                results.add(i, (x2b, rx1_2), "x1_2")


    # save final model
    vae.save_state()

    # reset tensorflow session and graph
    vae.sess.close()
    tf.reset_default_graph()

# save experimental results
Results.save(results)