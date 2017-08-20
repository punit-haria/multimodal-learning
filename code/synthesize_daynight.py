from data import DayNight
from training import Results
from results import image_plot


from runjoint import models


experiment_name = "daynight"

train_steps = 100000
save_steps = 10000

data = DayNight()

tracker = Results.load(experiment_name)

syntheses = ['reconstruct', 'sample']



for i in range(save_steps, train_steps+save_steps, save_steps):

    suffix = str(i)

    image_plot(tracker, models, data=data, suffix=suffix, syntheses=syntheses,
               n_rows=6, n_cols=6, n_pixels=0, spacing=0, model_type='joint', count=10)

