from training import Results
from results import curve_plot


from runvae import experiment_name, parms



tracker = Results.load(experiment_name)

curve_plot(tracker, curve_name='test_lower_bound', curve_label=None, axis=None,
           xlab='# Samples Evaluated', ylab='Nats')

curve_plot(tracker, curve_name='train_lower_bound', curve_label=None, axis=None,
           xlab='# Samples Evaluated', ylab='Nats')




