import numpy as np
import tensorflow as tf

from models.vae import VAE, VAECNN
from data import MNIST

from results import Results


# store experimental results
results = Results('experiment_mnist_cnn')


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

    'train_steps': 2000,
    'plot_steps': 500,
    'n_plots': 10
}


models = {
    'vae': VAE
    #'vae_translate': VAETranslate,
    #'vae_cnn': VAECNN
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
        x, _ = mnist.sample(parms['batch_size'], dtype='train', binarize=True)

        # training step
        bound = vae.train(x)

        # save results
        results.add(i, bound, "training_curve")

        if i % 25 == 0:
            print("At iteration ", i)

            # test minibatch
            x, _ = mnist.sample(1000, dtype='test', binarize=False)

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