
VERSION = 'V4.7'

import numpy
import matplotlib.pyplot as plt

from q_systems import SpinSystem
from qmcmc_vqe_classes_temperature import * # meglio usare temperature ma gli devi prima modificare run_random_qmcmc

# defining spin system and setting up qmcmc runner (values from IBM paper)
n_spins = 4
T = 0.1
#
ansatz = IBM_Ansatz  # do not put () here
# initializing dataframes to save data
sgap_df = DataFrame()
sgap_array = numpy.array([])
# running several simulations
instances_number = 50
mc_steps = int(500)
#
for instance in tqdm(range(instances_number)):
    #
    model_instance = IsingModel_2D(n_spins, random=True, nearest_neigh=False)
    J = model_instance.J
    h = model_instance.h
    #
    spin_system = SpinSystem(n_spins, T, J, h)  # probabilmente va rimosso
    # sampling a random initial value for the params
    params_bounds = {'gamma': (0.25, 0.6), 'tau': (1, 10)}
    # initializing optimizer class
    qmcmc_runner = QMCMC_Runner(spin_system, ansatz)
    # defining parameters initial guess (devi fare in modo che si adatti a diverso numero di params)
    params_string = '_'
    for param_name, bounds in params_bounds.items():
        params_string += param_name + f'_{round(bounds[0], 3)}_{round(bounds[1], 3)}_'
    #
    core_str = f'RAND_AVG_{instances_number}_npins_{n_spins}_qmcmc_{VERSION}' + params_string + \
               f'mc_steps_{mc_steps}_T_{T}_A_' + ansatz.name + '_mod_' + model_instance.name
    #
    sgap_array = qmcmc_runner.run_random_qmcmc(mc_steps, sgap_array, params_bounds=params_bounds)
#
sgap_df = sgap_df.append({'sgap mean': sgap_array.mean(), 'sgap std': sgap_array.std()}, ignore_index=True)
# print(sgap_df)
# saving the data as csv file
csv_name = 'data_csv_' + core_str
sgap_df.to_csv('./simulations_results/random_params/SG_' + csv_name + '.csv', encoding='utf-8')
print('\nsaved data to csv file: ' + csv_name + '\n')

# plotting the results TODO: CLASS FOR PLOTTING
sg_mean = numpy.full(shape=5, fill_value=sgap_df['sgap mean'])
sg_std = numpy.full(shape=5, fill_value=sgap_df['sgap std'])
# printing plots
figure, axis = plt.subplots((1), figsize=(12, 11), dpi=100)
figure.tight_layout(h_pad=2, w_pad=4)  # distances between subplots
# spectral gap
axis.plot(range(sg_mean.size), sg_mean, color='khaki', label='Random params',
               linestyle='-', lw=3)  # marker='o', markersize=10
axis.fill_between(range(sg_mean.size), sg_mean-sg_std, sg_mean+sg_std, alpha=0.3,
                     edgecolor='khaki', facecolor='khaki', linewidth=1)
axis.text(-0.3, + 1.2, '(a)' + f'   $n = {n_spins}$', fontsize = 30, transform=axis.transAxes, fontweight='bold')
axis.grid(linestyle='--')
axis.set_xticklabels([])
axis.set_ylabel('Spectral gap $\delta$', fontsize=20, labelpad=10)
axis.tick_params(labelsize=15, axis='both', which='major', pad=10, width=2, length=10)
axis.legend(fontsize=15)
# axis[0].set_title('Spectral gap $\delta$')
axis.set_ylim(0, 1)
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
png_name = './simulations_plots/random_params/plot_' + core_str + '.png'
figure.savefig(png_name, bbox_inches='tight')