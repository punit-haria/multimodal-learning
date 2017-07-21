from models.pixel_cnn import PixelCNN
from data import MNIST
from training import train, Results



tracker = Results('trial_pixelcnn')

models = {
    'PixelCNN': PixelCNN
}


# parameters
parms = {
    # basic parameters
    'height': 28,
    'width': 28,
    'n_channels': 1,

    # network parameters
    'n_feature_maps': 32,

    # autoregressive model parameters
    'n_pixelcnn_layers': 15,
    'conditional': True,
    'concat': True,

    # train/test parameters
    'learning_rate': 0.002,
    'batch_size': 16,
    'test_sample_size': 1000,
    'n_conditional_pixels': 300,
    'train_steps': 20000,
    'test_steps': 50,
    'save_steps': 5000
}



# data
mnist = MNIST()


for name, model in models.items():
    train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)




