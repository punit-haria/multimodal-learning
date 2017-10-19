from data import MSCOCO
from training import Results
from results import coco_plot

from runmscoco import models


experiment_name = "coco"

train_steps = 30000
save_steps = 3000

data = MSCOCO(60000)

tracker = Results.load(experiment_name)


for i in range(save_steps, train_steps+save_steps, save_steps):

    coco_plot(tracker, models, data=data, n_rows=5, n_cols=5, repetitions=1, train_steps=str(i))



