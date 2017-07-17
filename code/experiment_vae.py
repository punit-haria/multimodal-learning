import numpy as np
import tensorflow as tf

from models.vae import VAE, VAECNN
from data import MNIST

from results import Results


# store experimental results
results = Results('experiment_vae_pixel_cnn')


# parameters
parms = {
    'n_z': 100,
    'learning_rate': 0.002,
    'batch_size': 250,

    'n_channels': 1,
    'n_pixelcnn_layers': 3,

    'train_steps': 20000,
    'plot_steps': 4000,
    'test_steps': 50,
    'n_plots': 18
}


models = {
    #'VAE': VAE
    'VAE_CNN': VAECNN
}

# data
mnist = MNIST()


# tests
alphas = [0.125, 0.5, 1]
pixel_layers = [2, 3, 4]
learning_rates = [0.002, 0.0002]

for n_layers in pixel_layers:
    for lr in learning_rates:
        for alpha in alphas:

            parms['n_pixelcnn_layers'] = n_layers
            parms['learning_rate'] = lr
            parms['alpha'] = -alpha


            for name, model in models.items():

                name = name + '_layers_' + str(n_layers) + '_lrate_' + str(lr) + '_alpha_' + str(alpha)

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
                    bound, loss, reconstruction, penalty = vae.train(x)

                    # save results
                    results.add(i, bound, "train_lower_bound")
                    results.add(i, loss, "train_loss")
                    results.add(i, reconstruction, "train_reconstruction")
                    results.add(i, penalty, "train_penalty")

                    if i % parms['test_steps'] == 0:
                        print("At iteration ", i, flush=True)

                        # test minibatch
                        x = mnist.sample(1000, dtype='test', binarize=False)[0]

                        # test model
                        bound, loss, reconstruction, penalty = vae.test(x)

                        # save results
                        results.add(i, bound, "test_lower_bound")
                        results.add(i, loss, "test_loss")
                        results.add(i, reconstruction, "test_reconstruction")
                        results.add(i, penalty, "test_penalty")

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

                # save intermediate experimental results
                Results.save(results)

# save experimental results
Results.save(results)



