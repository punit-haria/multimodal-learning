from data import ColouredStratifiedMNIST
from training import Results
from results import image_plot


from runjoint import models


experiment_name = "discrete_colored_final"

train_steps = 50000
save_steps = 50000

data = ColouredStratifiedMNIST(2000)

tracker = Results.load(experiment_name)

syntheses = ['repeated_synth']#, 'sample', 'reconstruct', 'fix_latents']
#syntheses = ['repeated_synth', 'fix_latents']


for i in range(save_steps, train_steps+save_steps, save_steps):

    suffix = str(i)

    image_plot(tracker, models, data=data, suffix=suffix, syntheses=syntheses,
               n_rows=8, n_cols=4, n_pixels=0, spacing=0, model_type='joint', count=20)

