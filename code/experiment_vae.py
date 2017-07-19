import numpy as np
import tensorflow as tf

from models.vae import VAE, VAECNN
from data import MNIST

from results import Results


# store experimental results
results = Results('experiment_vae_conditional_pixelcnn')


# parameters
parms = {
    'n_z': 200,
    'learning_rate': 0.002,
    'batch_size': 64,

    'n_channels': 1,
    'n_pixelcnn_layers': 2,
    'concat': True,
    'n_pixels': 300,

    'train_steps': 10000,
    'plot_steps': 5000,
    'test_steps': 50,
    'n_plots': 18
}


models = {
    'VAE_conditional_pixelCNN': VAE
    #'VAE_CNN': VAECNN
}

# data
mnist = MNIST()

batch_sizes = [16, 64, 256]
concat = [True, False]
layers = [2, 6]


for conc in concat:
    for lays in layers:
        for bs in batch_sizes:

            parms['concat'] = conc
            parms['n_pixelcnn_layers'] = lays
            parms['batch_size'] = bs

            for name, model in models.items():

                name = name + '_concat_' + str(conc) + '_layers_' + str(lays) + '_batch_size_' + str(bs)

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
                        x = mnist.sample(1000, dtype='test', binarize=True)[0]

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
                            #rx_probs = vae.reconstruct(xb)
                            rx_probs = vae.autoregressive_reconstruct(xb, parms['n_pixels'])

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



