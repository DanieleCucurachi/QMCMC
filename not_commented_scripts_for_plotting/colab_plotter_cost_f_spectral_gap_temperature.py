
VERSION = 'V4.7'

# importing libraries
import numpy
import scipy
import matplotlib.pyplot as plt
from q_systems import SpinSystem
from qmcmc_vqe_classes_temperature import *
from matplotlib import rcParams

# modifying font  # fontname='Liberation Serif'
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman']
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.size'] = 12

# CAMBIA A MANO VQE CALSSES SE NON VUOI LA NORMALIZAZIONE SULLA ACF (when t fixed)

# defining spin system
n_spins = 3
T = 10  # TEMPERATURE
# numpy.random.seed(67623)
model_instance = IsingModel_2D(n_spins, random=True)  # , nearest_neigh=True, spins_grid=(3, 2)
J = model_instance.J
h = model_instance.h
spin_system = SpinSystem(n_spins, T, J, h)

#
ansatz = IBM_Ansatz  # do not put () here
#
mc_length = 10  # n_spins**2
average_over = 1
discard = n_spins*1e3
lag = 2
#
cost_f_choice = 'ACF'
observable = 'energy'
optimization_approach = 'same_start_mc'  # I want every cost func evslusiton to start from the same state
#
qmcmc_opt = QMCMC_Optimizer(spin_system, ansatz, mc_length, average_over=average_over,
              cost_f_choice=cost_f_choice, optimization_approach=optimization_approach,
              verbose=False, initial_transient=discard, observable=observable, lag=lag)
# run start from the same state
initial_state = qmcmc_opt.current_state 

# gathering cost f and spectral gap data
resolution = 10
gammas = numpy.linspace(0.07, 1, resolution)
taus = numpy.linspace(1, 10, resolution)
#
cost_f = numpy.zeros(shape=(resolution, resolution), dtype=float)
spectral_gap = numpy.zeros(shape=(resolution, resolution), dtype=float)
#
for i, gamma in tqdm(enumerate(gammas)):
    for j, tau in enumerate(taus):
        # defining params array
        params = numpy.array([gamma, tau])
        # calculating cost function
        cost_f[j][i] = qmcmc_opt(params, initial_state)  # check this is calculated in the right way with respect to how python plot with meshgrid 
        # calculating spectral gap
        spectral_gap[j][i] = qmcmc_opt.calculate_sgap(params)
#
X, Y = numpy.meshgrid(gammas, taus)

#
core_str = f'_qmcmc_{VERSION}' + f'_plots_gamma_{gammas[0]}_{gammas[-1]}_tau_{taus[0]}_{taus[-1]}_res_{resolution}_' + \
           'cost_f_' + cost_f_choice + '_' + f'mc_length_{mc_length}_T_{T}_npins_{n_spins}_av_' + \
           f'{average_over}_opt_' + ansatz.name + '_mod_' + model_instance.name
if cost_f_choice == 'ACF':
    if isinstance(lag, dict):
        lag = f"optmz_{lag['lag']}_{lag['acf_noise']}_{lag['lag_scale']}"
    core_str += f'_discard_{discard}_lag_{lag}_obs_' + observable

# saving the data in csv file
cf_data_string = './colab/data/temp_data_cf_' + core_str + '.csv'
numpy.savetxt(cf_data_string, cost_f, delimiter=",")
sg_data_string = './colab/data/temp_data_sg_' + core_str + '.csv'
numpy.savetxt(sg_data_string, spectral_gap, delimiter=",")


# USA COLAB PER STAMPARE LA HEATMAP DEL COUPLING PER GLI SPINS
# saving plots
xticklabels=[numpy.round(gammas, decimals=2)[i] if i%(resolution//5)==0 or i+1==resolution else None for i in range(gammas.size)]
yticklabels=[numpy.round(taus, decimals=1)[i] if i%(resolution//5)==0 or i+1==resolution else None for i in range(taus.size)]
# plotting colormap cost function
plt.figure(figsize=(15,12), dpi=300)
# t integral
# s = sns.heatmap(cost_f, square=True, annot=False, cbar=True, cmap='viridis', xticklabels=xticklabels, yticklabels=yticklabels, norm=matplotlib.colors.LogNorm(vmin=1, vmax=1000))   # viridis coolwarm
# s = sns.heatmap(cost_f, square=True, annot=False, cbar=True, cmap='viridis', xticklabels=xticklabels, yticklabels=yticklabels, norm=matplotlib.colors.LogNorm())
# fixed t
s = sns.heatmap(cost_f, square=True, annot=False, cbar=True, cmap='viridis', xticklabels=xticklabels, yticklabels=yticklabels)  # vmin=0, vmax=200
s.set_xlabel('$\gamma$', fontsize=55, labelpad=40)
s.set_ylabel('$t$', fontsize=55, labelpad=40)
s.tick_params(labelsize=25, axis='both', which='major', pad=20)
s.figure.axes[1].tick_params(labelsize=25, width=2, length=10)
s.collections[0].colorbar.set_label('spectral gap $\delta$', fontsize=40, labelpad=30)  # , fontname='Liberation Serif'
# plt.yticks(rotation=0)
# plt.xticks(rotation=0)  
plt.savefig('./colab/plots/temp_colmap_cost_f_' + core_str + '.png', bbox_inches='tight')
# plotting colormap spectral gap
plt.figure(figsize=(15,12), dpi=300)
s = sns.heatmap(spectral_gap, square=True, annot=False, cbar=True, cmap='Spectral', xticklabels=xticklabels, yticklabels=yticklabels)
s.set_xlabel('$\gamma$', fontsize=55, labelpad=40)
s.set_ylabel('$t$', fontsize=55, labelpad=40)
s.tick_params(labelsize=25, axis='both', which='major', pad=20) # width=2, length=10
s.figure.axes[1].tick_params(labelsize=25, width=2, length=10)
s.collections[0].colorbar.set_label('spectral gap $\delta$', fontname='Liberation Serif', fontsize=40, labelpad=30)
# plt.yticks(rotation=0)
# plt.xticks(rotation=0)  
plt.savefig('./colab/plots/temp_colmap_sgap_' + core_str + '.png', bbox_inches='tight')