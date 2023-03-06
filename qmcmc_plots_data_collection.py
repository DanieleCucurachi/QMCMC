
VERSION = 'V4.4'

import numpy
import scipy
import matplotlib.pyplot as plt

from q_systems import SpinSystem
from qmcmc_vqe_classes import *

# defining spin system and setting up qmcmc runner (values from IBM paper)
n_spins = 3
T = 10
# numpy.random.seed(630201)
model_instance = IsingModel_2D(n_spins, random=True, nearest_neigh=False)
J = model_instance.J
h = model_instance.h
#
spin_system = SpinSystem(n_spins, T, J, h)  # probabilmente va rimosso
ansatz = IBM_Ansatz  # do not put () here

# initializing dataframes to save data
sg_df = DataFrame()
cf_df = DataFrame()
# running several simulations over the same model instance
simulations_number = 10
for simul in tqdm(range(simulations_number)):
    #
    ansatz = IBM_Ansatz  # do not put () here
    #
    mc_length = 1000  # n_spins**2
    discard = n_spins*1e3
    lag = 4
    average_over = 10
    # sampling a random initial value for the params
    params_bounds = {'gamma': (0.1, 0.25), 'tau': (2, 10)}
    params_dict = {'gamma': numpy.random.uniform(low=params_bounds['gamma'][0], high=params_bounds['gamma'][1], size=None),
                   'tau': numpy.random.uniform(low=params_bounds['tau'][0], high=params_bounds['tau'][1], size=None)}  # RANDOM
    maxiter = 200 * len(params_dict.keys())
    # defining optimizer specs
    cost_f_choice = 'ACF'
    observable = 'energy'
    optimization_approach = 'same_start_mc'
    # initializing optimizer class
    qmcmc_optimizer = QMCMC_Optimizer(spin_system, ansatz, mc_length, average_over=average_over,
                       cost_f_choice=cost_f_choice, optimization_approach=optimization_approach,
                       initial_transient=discard, observable=observable, lag=lag)
    # defining parameters initial guess (devi fare in modo che si adattia diverso numero di params)
    params_guess = numpy.fromiter(params_dict.values(), dtype=float)  # , dtype=float
    params_string = '_'
    for param_name, bounds in params_bounds.items():
        params_string += param_name + f'_{round(bounds[0], 3)}_{round(bounds[1], 3)}_'
    # including the initial params values and corresponding spectral gap value
    cost_f = qmcmc_optimizer(params_guess, qmcmc_optimizer.current_state)
    qmcmc_optimizer.get_save_results(params=params_guess, cost_f=cost_f)
    # defining scipy optimizer specs
    bnds = ((0.1, 1), (1, 10))
    optimizer ='Nelder-Mead'
    # running optimization algorithm
    while data_to_collect(qmcmc_optimizer, max_iteration=35e3): 
    
        args = (qmcmc_optimizer.current_state)
        results = scipy.optimize.minimize(qmcmc_optimizer, x0=params_guess, args=args, 
                      method=optimizer, bounds=bnds, options = {'maxiter': maxiter})
        params_guess = results.x
        qmcmc_optimizer.get_save_results(results=results)
    # printing current result
    print(f'\n\n------------ simulation # {simul + 1} ------------\n')
    # 
    # TODO: MA LA EPSILON?
    sg_df[f'spectral gap {simul}'] = qmcmc_optimizer.db['spectral gap']
    cf_df[f'cost f {simul}'] = qmcmc_optimizer.db['cost f']
  
# computing mean and variance over the simulations results
sg_df['mean'] = sg_df.mean(axis=1)
sg_df['std'] = sg_df.std(axis=1)
cf_df['mean'] = cf_df.mean(axis=1)
cf_df['std'] = cf_df.std(axis=1)
# saving the data
if cost_f_choice == 'ACF':
    core_str = f'AVRG_qmcmc_{VERSION}' + params_string + 'cost_f_' + cost_f_choice + '_' + f'_discard_{discard}_lag_{lag}_obs_' + observable + '_' + \
          f'mc_length_{mc_length}_T_{T}_npins_{n_spins}_iter_{qmcmc_optimizer.iteration}_maxiter_{maxiter}_av_' + \
          f'{average_over}_opt_' + optimizer + '_a_' + optimization_approach + '_A_' + ansatz.name + '_mod_' + model_instance.name
else:
    core_str = f'AVRG_qmcmc_{VERSION}' + params_string + 'cost_f_' + cost_f_choice + '_' + \
          f'mc_length_{mc_length}_T_{T}_npins_{n_spins}_iter_{qmcmc_optimizer.iteration}_maxiter_{maxiter}_av_' + \
          f'{average_over}_opt_' + optimizer + '_a_' + optimization_approach + '_A_' + ansatz.name + '_mod_' + model_instance.name
# as csv file
csv_name = 'data_csv_' + core_str + '.csv'
sg_df.to_csv('./simulations_results/SG_' + csv_name, encoding='utf-8')
cf_df.to_csv('./simulations_results/CF_' + csv_name, encoding='utf-8')
print('\nsaved data to csv file: ' + csv_name + '\n')

# plotting the results TODO: CLASS FOR PLOTTING
sg_mean = numpy.array(sg_df['mean'])
sg_std = numpy.array(sg_df['std'])
cf_mean = numpy.array(cf_df['mean'])  # TODO: normalize so that can be compared to the C calculated numerically (maybe not possible)
cf_std = numpy.array(cf_df['std'])
figure, axis = plt.subplots((3), figsize=(12, 11), dpi=100)
figure.tight_layout(h_pad=2, w_pad=4)  # distances between subplots
# printin plots
if cost_f_choice == 'ACF':
    # spectral gap
    axis[0].plot(range(sg_mean.size), sg_mean, color='blue', label='Optimization with $' + cost_f_choice + '$',
                   linestyle='-', lw=3)  # marker='o', markersize=10
    axis[0].fill_between(range(sg_mean.size), sg_mean-sg_std, sg_mean+sg_std, alpha=0.3,
                         edgecolor='blue', facecolor='blue', linewidth=1)
    axis[0].text(-0.3, + 1.2, '(a)' + f'   $n = {n_spins}$', fontsize = 30, transform=axis[0].transAxes, fontweight='bold')
    axis[0].grid(linestyle='--')
    axis[0].set_xticklabels([])
    axis[0].set_ylabel('Spectral gap $\delta$', fontsize=20, labelpad=10)
    axis[0].tick_params(labelsize=15, axis='both', which='major', pad=10, width=2, length=10)
    axis[0].legend(fontsize=15)
    # axis[0].set_title('Spectral gap $\delta$')
    # axis[0].set_ylim(0, 1)
    for ax in ['top','bottom','left','right']:
        axis[0].spines[ax].set_linewidth(2)
    # cost function  # TODO: normalize cost fucntion such that u can compare with axis
    axis[1].plot(range(cf_mean.size), cf_mean, color='orange',
                      label='$ACF$', linestyle='-', lw=3)  # , markersize=10, marker='o'
    axis[1].fill_between(range(cf_mean.size), cf_mean-cf_std, cf_mean+cf_std, alpha=0.3,
                         edgecolor='orange', facecolor='orange', linewidth=1)
    axis[1].grid(linestyle='--')
    axis[1].set_xticklabels([])
    axis[1].set_ylabel('Cost function $' + cost_f_choice + '$', fontsize=20, labelpad=10)
    axis[1].tick_params(labelsize=15, axis='both', which='major', pad=10, width=2, length=10)
    axis[1].legend(fontsize=15)
    for ax in ['top','bottom','left','right']:
        axis[1].spines[ax].set_linewidth(2)
    # axis[1].set_title('Cost function $' + cost_f_choice + '$')
    # autocorrelation function
    if not isinstance(lag, str):  # in case it's not? what can we plot?
        if isinstance(lag, numpy.ndarray):
            lag = lag[0]
        acf = (1 - sg_mean)**lag
        acf_std = lag*(1-sg_mean)**(lag-1) * sg_std  # calculated with error propagation
        axis[2].plot(range(acf.size), acf, color='red',
                          label='$e^{-\\frac{t}{\\tau}}=\lambda_{_{SLEM}}^t$', linestyle='-', lw=3)  # marker='o', markersize=10
        axis[2].fill_between(range(acf.size), acf-acf_std, acf+acf_std, alpha=0.3,
                         edgecolor='blue', facecolor='red', linewidth=1)
        axis[2].grid(linestyle='--')
        axis[2].set_xlabel('Optimization steps', fontsize=20, labelpad=10)
        axis[2].tick_params(labelsize=15, axis='both', which='major', pad=10, width=2, length=10)
        # here u r taking for granted that energy is the f that decorrelates the slowest, might not be true TODO: specifica questa cosa meglio
        axis[2].set_ylabel('Autocorrelation Func $C(t) \sim e^{-\\frac{t}{\\tau}}=\lambda_{_{SLEM}}^t$', fontsize=20, labelpad=10)
        axis[2].legend(fontsize=15)
        for ax in ['top','bottom','left','right']:
            axis[2].spines[ax].set_linewidth(2)
        # axis[2].set_title('Autocorrelation Function $C(t) \sim e^{-\\frac{t}{\\tau}}=\lambda^t$')
    #
elif cost_f_choice == 'L':  # TODO: VA CAMBIATO ANCORA TUTTO QUA SOTTO, DEVI COPIARE CIO CHE HAI FATTO PER I PLOT SOPRA
    pass
    # spectral gap
    # axis[0].errorbar(sg_mean.size, sg_mean, sg_std, marker='o', color='blue',
    #                   label='$\delta$', linestyle='-')
    # axis[0].grid(linestyle='--')
    # axis[0].set_xlabel('Optimization steps')
    # axis[0].set_title('Spectral gap $\delta$')
    # axis[0].set_ylim(0, 1)
    # # cost function
    # axis[1].errorbar(cf_mean.size, cf_mean, cf_std, marker='o', color='red',
    #                   label='Cost f', linestyle='-')
    # axis[1].grid(linestyle='--')
    # axis[1].set_xlabel('Optimization steps')
    # axis[1].set_title('Cost function $' + cost_f_choice + '$')
    #
    # # explored states 
    # exp_states = qmcmc_optimizer.full_explored_states
    # axis[0][1].bar(range(exp_states.size), exp_states, color='green')
    # axis[0][1].grid(linestyle='--')
    # axis[0][1].set_xlabel('Spin configurations')
    # axis[0][1].set_title('Frequency during optimization')
    #
    # # total variational distance  # TODO
    # eps = numpy.array(qmcmc_optimizer.db['epsilon'])
    # axis[2].plot(range(eps.size), eps, marker='o', color='orange', label='$\epsilon$',
    #                  linestyle='-')
    # axis[2].grid(linestyle='--')
    # axis[2].set_xlabel('Optimization steps')
    # axis[2].set_title('Total variational distance $\epsilon$')
    #
#
figure.align_ylabels(axis[:])
# saving the plot as png file
png_name = './simulations_plots/' + cost_f_choice + '/' + 'plot_' + core_str + '.png'
figure.savefig(png_name, bbox_inches='tight')