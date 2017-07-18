import tensorflow as tf

from models.pixel_cnn import PixelCNN
from data import MNIST

from results import Results


# store experimental results
results = Results('experiment_pixelcnn')


# parameters
parms = {
    'learning_rate': 0.002,
    'batch_size': 200,

    'n_channels': 1,
    'n_layers': 3,

    'train_steps': 1000,
    'plot_steps': 1000,
    'test_steps': 50,
    'n_plots': 18,
    'n_pixels': 300
}


models = {
    'PixelCNN': PixelCNN
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
        loss = vae.train(x)

        # save results
        results.add(i, loss, "train_loss")

        if i % parms['test_steps'] == 0:
            print("At iteration ", i, flush=True)

            # test minibatch
            x = mnist.sample(1000, dtype='test', binarize=False)[0]

            # test model
            loss = vae.test(x)

            # save results
            results.add(i, loss, "test_loss")

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



