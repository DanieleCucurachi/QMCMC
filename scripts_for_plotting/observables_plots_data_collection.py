
VERSION = 'V4.7'

import numpy
import matplotlib.pyplot as plt

from q_systems import SpinSystem
from qmcmc_vqe_classes import *

# defining spin system and setting up qmcmc runner (values from IBM paper)
n_spins = 6
T = 10
#
ansatz = IBM_Ansatz  # do not put () here
#
en_df = DataFrame()
mag_df = DataFrame()
# running several simulations
instances_number = 5
simulations_number = 5
mc_steps = int(3000)
#
for instance in tqdm(range(instances_number)):
    #
    model_instance = IsingModel_2D(n_spins, random=True, nearest_neigh=False)
    J = model_instance.J
    h = model_instance.h
    #
    spin_system = SpinSystem(n_spins, T, J, h)  # probabilmente va rimosso
    #
    for simulation in range(simulations_number):
        #
        idx = instances_number*instance + simulation
        # initializing optimizer class
        qmcmc_runner = QMCMC_Runner(spin_system, ansatz)
        # sampling a random initial value for the params
        params_bounds = {'gamma': (0.2, 0.6), 'tau': (2, 20)}  #TODO: FIXED PARAMS??????
        params_string = '_'
        for param_name, bounds in params_bounds.items():
            params_string += param_name + f'_{round(bounds[0], 3)}_{round(bounds[1], 3)}_'
        #
        core_str = f'OBS_CON_AVG_{instances_number}_npins_{n_spins}_qmcmc_{VERSION}' + params_string + \
                   f'mc_steps_{mc_steps}_T_{T}_A_' + ansatz.name + '_mod_' + model_instance.name
        #
        en_df, mag_df = qmcmc_runner.observables_convergence_check(mc_steps, en_df, mag_df, idx, params_bounds=params_bounds)
        #
        # core_str = 'optimal'
        # fixed_params = [0.63, 5]
        # en_df, mag_df = qmcmc_runner.observables_convergence_check(mc_steps, en_df, mag_df, idx, fixed_params=fixed_params)
        #
# computing mean and variance over the simulations results
en_df['mean'] = en_df.mean(axis=1)
en_df['std'] = en_df.std(axis=1)
mag_df['mean'] = mag_df.mean(axis=1)
mag_df['std'] = mag_df.std(axis=1)
# saving the data as csv file
csv_name = 'data_csv_' + core_str
en_df.to_csv('./simulations_results/observables/EN_' + csv_name + '.csv', encoding='utf-8')
mag_df.to_csv('./simulations_results/observables/MAG_' + csv_name + '.csv', encoding='utf-8')
print('\nsaved data to csv file: ' + csv_name + '\n')

# plotting the results TODO: CLASS FOR PLOTTING
en_mean = numpy.array(en_df['mean'])
en_std = numpy.array(en_df['std'])
mag_mean = numpy.array(mag_df['mean'])
mag_std = numpy.array(mag_df['std'])
# printing plots
figure, axis = plt.subplots((1), figsize=(12, 11), dpi=100)
figure.tight_layout(h_pad=2, w_pad=4)  # distances between subplots
# energy
axis.plot(range(en_mean.size), en_mean, color='khaki', label='$E$',
               linestyle='-', lw=3)  # marker='o', markersize=10
axis.fill_between(range(en_mean.size), en_mean-en_std, en_mean+en_std, alpha=0.3,
                     edgecolor='khaki', facecolor='khaki', linewidth=1)
#
axis.plot(range(mag_mean.size), mag_mean, color='blue', label='$M$',
               linestyle='-', lw=3)  # marker='o', markersize=10
axis.fill_between(range(mag_mean.size), mag_mean-mag_std, mag_mean+mag_std, alpha=0.3,
                     edgecolor='blue', facecolor='blue', linewidth=1)
#
axis.text(-0.3, + 1.2, '(a)' + f'   $n = {n_spins}$', fontsize = 30, transform=axis.transAxes, fontweight='bold')
axis.grid(linestyle='--')
axis.set_xticklabels([])
axis.set_ylabel('Observables', fontsize=20, labelpad=10)
axis.tick_params(labelsize=15, axis='both', which='major', pad=10, width=2, length=10)
axis.legend(fontsize=15)
# axis[0].set_title('Spectral gap $\delta$')
# axis.set_ylim(0, 1)
for ax in ['top','bottom','left','right']:
    axis.spines[ax].set_linewidth(2)
# # 
# axis[1].plot(range(en_mean.size), en_mean, color='blue',
#                   label='$E$', linestyle='-', lw=3)  # , markersize=10, marker='o'
# axis[1].fill_between(range(en_mean.size), en_mean-en_std, en_mean+en_std, alpha=0.3,
#                      edgecolor='blue', facecolor='blue', linewidth=1)
# #
# axis[1].plot(range(mag_mean.size), mag_mean, color='orange',
#                   label='$M$', linestyle='-', lw=3)  # , markersize=10, marker='o'
# axis[1].fill_between(range(mag_mean.size), mag_mean-mag_std, mag_mean+mag_std, alpha=0.3,
#                      edgecolor='orange', facecolor='orange', linewidth=1)
# #
# axis[1].grid(linestyle='--')
# axis[1].set_ylabel('Observables $E$ and $M$', fontsize=20, labelpad=10)
# axis[1].tick_params(labelsize=15, axis='both', which='major', pad=10, width=2, length=10)
# axis[1].legend(fontsize=15)
# for ax in ['top','bottom','left','right']:
#     axis[1].spines[ax].set_linewidth(2)
# axis[1].set_title('Cost function $' + cost_f_choice + '$')
#
figure.align_ylabels(axis)
# printing plots
plt.show()
# saving the plot as png file
png_name = './simulations_plots/observables/plot_' + core_str + '.png'
figure.savefig(png_name, bbox_inches='tight')