from training import Results
from results import curve_plot


from run_vae import experiment_name, parms



tracker = Results.load(experiment_name)

curve_plot(tracker, parms, curve_name='test_lower_bound', curve_label=None, axis=None,
           xlab='Steps', ylab='Nats')





