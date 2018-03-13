from training import Results

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

cm_choice = cm.Greys  # Greys_r
plt.style.use('ggplot')

'''

This script generates training and test curve plots for a given experiment.
 
'''



experiment_name = "discrete_colored_final"

tracker = Results.load(experiment_name)

curve_name='test_lower_bound_on_log_p_x_y'


legend='lower right'
legend_font=18
xlab='# Epochs'
ylab=r'$\log \ p(x,y) \ \geq$'


names = tracker.get_runs()

allowed_names = {
    'discrete_colored_final_cnn_small_nz_64_lr_0.001_fmaps_32_units_128_obj_joint_jointanneal_0.3': 'CNN-VAE',
    'discrete_colored_final_cnn_small_nz_64_lr_0.001_fmaps_32_units_128_obj_joint_ar_1_anneal_-0.25_jointanneal_0.3':
    'AR-CNN-VAE'
}


plt.figure(figsize=(12, 9))

labels = names
for label, name in zip(labels, names):

    if name not in allowed_names:
        continue
    else:
        label = allowed_names[name]

    trial = tracker.get(name)
    x, f = trial.get_series(curve_name)
    parms = trial.parameters

    bs = 31.25
    x = [_x / bs for _x in x]

    plt.plot(x, f, label=label, linewidth=2)

plt.axis([0, 2000, -2500, 0])

plt.legend(loc=legend, fontsize=legend_font)
plt.xlabel(xlab)
plt.ylabel(ylab)

plt.savefig('../plots/' + tracker.name + '_' + curve_name + '.png')
plt.close('all')