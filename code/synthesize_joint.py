from data import ColouredStratifiedMNIST
from training import Results
from results import image_plot


from runjoint import models


experiment_name = "ar_nf_colored_mnist"

train_steps = 60000
save_steps = 10000

data = ColouredStratifiedMNIST(5000)

tracker = Results.load(experiment_name)

syntheses = ['reconstruct', 'sample']



for i in range(save_steps, train_steps+save_steps, save_steps):

    suffix = str(i)

    image_plot(tracker, models, data=data, suffix=suffix, syntheses=syntheses,
               n_rows=8, n_cols=8, n_pixels=0, spacing=0, model_type='joint', count=4)

