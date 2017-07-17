import numpy as np
import tensorflow as tf

from models.vae import VAE, VAECNN
from data import MNIST

from results import Results


# store experimental results
results = Results('experiment_vae_mnist')


# parameters
parms = {
    'n_z': 50,
    'n_x': 784,
    'n_enc_units': 200,
    'learning_rate': 0.002,
    'batch_size': 250,

    'image_dim': [28, 28, 1],
    'filter_w': 3,
    'n_dec_units': 200,
    'n_dec_layers': 3,

    'train_steps': 500000,
    'plot_steps': 40000,
    'test_steps': 1000,
    'n_plots': 18
}


models = {
    'VAE': VAE
    #'vae_translate': VAETranslate,
    #'VAE_CNN': VAECNN
}

# data
mnist = MNIST()


for name, model in models.items():

    # load model
    print("Training Model: ", name, flush=True)
    vae = model(arguments=parms, name=name)

    # store next experimental run
    results.create_run(name)

    # train model
    for i in range(parms['train_steps'] + 1):

        # random minibatch
        x = mnist.sample(parms['batch_size'], dtype='train', binarize=True)[0]

        # training step
        curve, bound = vae.train(x)

        # save results
        results.add(i, curve, "training_curve")
        results.add(i, bound, "train_lower_bound")

        if i % parms['test_steps'] == 0:
            print("At iteration ", i)

            # test minibatch
            x = mnist.sample(1000, dtype='test', binarize=False)[0]

            # test model
            bound = vae.test(x)

            # save results
            results.add(i, bound, "test_lower_bound")

            # plot reconstructions
            if i % parms['plot_steps'] == 0:

                n_examples = parms['n_plots']
                xb = x[0:n_examples]
                rx_probs = vae.reconstruct(xb)

                # save reconstructions
                results.add(i, (xb, rx_probs), "x_reconstructed")

    # save final model
    vae.save_state()

    # reset tensorflow session and graph
    vae.sess.close()
    tf.reset_default_graph()

# save experimental results
Results.save(results)