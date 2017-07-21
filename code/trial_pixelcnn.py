from models.pixel_cnn import PixelCNN
from data import MNIST
from training import train, Results



tracker = Results('experiment_pixelcnn')

models = {
    'PixelCNN': PixelCNN
}

# parameters
parms = {
    'learning_rate': 0.002,
    'batch_size': 16,

    'n_channels': 1,
    'n_layers': 15,

    'train_steps': 20000,
    'plot_steps': 5000,
    'test_steps': 50,
    'n_plots': 18,
    'n_pixels': 300
}



# data
mnist = MNIST()


for name, model in models.items():
    train(name=name, model=model, parameters=parms, data=mnist, tracker=tracker)




