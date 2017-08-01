from data import MNIST, CIFAR, ColouredStratifiedMNIST
from training import Results
from results import image_plot


from runjoint import models


experiment_name = "joint"
suffix = None # None

data = ColouredStratifiedMNIST(1000)


tracker = Results.load(experiment_name)

syntheses = ['reconstruct']

image_plot(tracker, models, data=data, suffix=suffix, syntheses=syntheses,
           n_rows=8, n_cols=8, n_pixels=300, spacing=0, )
