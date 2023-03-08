
VERSION = 'V4.5'

import numpy
import scipy
import matplotlib.pyplot as plt

from q_systems import SpinSystem
from qmcmc_vqe_classes import *

# defining spin system and setting up qmcmc runner (values from IBM paper)
n_spins = 5
T = 10
# numpy.random.seed(630201)
model_instance = IsingModel_2D(n_spins, random=True)
J = model_instance.J
h = model_instance.h
#
spin_system = SpinSystem(n_spins, T, J, h)  # probabilmente va rimosso
ansatz = IBM_Ansatz  # do not put () here
#
mc_length = 2000  # n_spins**2
discard = n_spins*1e3
lag = 'integral'
average_over = 1
params_dict = {'gamma': 0.16, 'tau': 2}
maxiter = 200 * len(params_dict.keys())
#
cost_f_choice = 'ACF'
observable = 'energy'
optimization_approach = 'concatenated_mc'
#
qmcmc_optimizer = QMCMC_Optimizer(spin_system, ansatz, mc_length, average_over=average_over,
                   cost_f_choice=cost_f_choice, optimization_approach=optimization_approach,
                   verbose=True, initial_transient=discard, observable=observable, lag=lag)
# defining parameters initial guess (devi fare in modo che si adattia diverso numero di params)
params_guess = numpy.fromiter(params_dict.values(), dtype=float)  # , dtype=float
params_string = '_'
for param_name, value in params_dict.items():
    params_string += param_name + f'_{round(value, 3)}_'
# include the initial params values and corresponding spectral gap value
cost_f = qmcmc_optimizer(params_guess, qmcmc_optimizer.current_state)
qmcmc_optimizer.get_save_results(params=params_guess, cost_f=cost_f)

# defining scipy optimizer settings
bnds = ((0.1, 1), (1, 10))
optimizer ='Nelder-Mead'
# initial_simplex = numpy.array([[0.16, 2],
#                                [0.5, 2],
#                                [0.8, 7]])# array_like of shape (N + 1, N)
# fatol =  # The difference of function values at the vertices of the simplex is at most fatol
# xatol =  # The size of the simplex is at most xatol

# 
core_str = f'qmcmc_{VERSION}' + params_string + 'cost_f_' + cost_f_choice + '_' + \
           f'mc_length_{mc_length}_T_{T}_npins_{n_spins}_maxiter_{maxiter}_av_' + \
           f'{average_over}_opt_' + optimizer + '_a_' + optimization_approach + '_A_' + \
           ansatz.name + '_mod_' + model_instance.name
if cost_f_choice == 'ACF':
    if isinstance(lag, dict):
        lag = lag['lag']
    core_str += f'_discard_{discard}_lag_{lag}_obs_' + observable
#
print('\nsimulation: ' + core_str + '\n')

while data_to_collect(qmcmc_optimizer, max_iteration=30e3): 

    #
    args = (qmcmc_optimizer.current_state)
    #
    results = scipy.optimize.minimize(qmcmc_optimizer, x0=params_guess, args=args, 
                  method=optimizer, bounds=bnds, options = {'maxiter': maxiter, 
                  'adaptive': True if params_guess.size > 3 else False, 'initial_simplex': None})
    # The tuple passed as args will be passed as *args to the objective function (*args is a tuple, **kwargs is a dictionary)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html to see how to provide args correctly 
    params_guess = results.x
    #
    if isinstance(lag, dict):
        qmcmc_optimizer.optimize_lag()
    #
    qmcmc_optimizer.get_save_results(results=results)

print('\nOptimization terminated with:', qmcmc_optimizer.iteration, 'cost function evaluations\n')

# saving the results
csv_name = './simulations_results/' + core_str + f'_iter_{qmcmc_optimizer.iteration}_' + '.csv'
qmcmc_optimizer.db.to_csv(csv_name, encoding='utf-8')
print('\nsaved data to csv file: ' + csv_name + '\n')

# plotting the results MAYBE HERE CREATE A CLASS FOR PLOTTING
#
spectral_gap = numpy.array(qmcmc_optimizer.db['spectral gap'])
cost_function = numpy.array(qmcmc_optimizer.db['cost f'])
figure, axis = plt.subplots(2,2)  # , figsize=(12, 11), dpi=700
figure.tight_layout(h_pad=6, w_pad=4)  # distances between subplots
#
if cost_f_choice == 'ACF':
    # spectral gap
    axis[0][0].plot(range(spectral_gap.size), spectral_gap, marker='o', color='blue',
                      label='$\delta$', linestyle='-')
    axis[0][0].grid(linestyle='--')
    axis[0][0].set_xlabel('Optimization steps')
    axis[0][0].set_title('Spectral gap $\delta$')
    axis[0][0].set_ylim(0, 1)
    # cost function
    axis[1][0].plot(range(cost_function.size), cost_function, marker='o', color='red',
                      label='Cost f', linestyle='-')
    axis[1][0].grid(linestyle='--')
    axis[1][0].set_xlabel('Optimization steps')
    axis[1][0].set_title('Cost function $' + cost_f_choice + '$')
    if not isinstance(lag, str):  # in case it's not? what can we plot?
        if isinstance(lag, dict):
            lag = lag['lag']
        # autocorrelation time
        axis[1][1].plot(range(spectral_gap.size), (1 - spectral_gap)**lag, marker='o', color='orange',
                          label='$C(t) \sim e^{-t/\\tau}=\lambda^t$', linestyle='-')
        axis[1][1].grid(linestyle='--')
        axis[1][1].set_xlabel('Optimization steps')
        axis[1][1].set_title('ACF $C(t) \sim e^{-t/\\tau}=\lambda^t$')
    # TODO: if you use integral you should calcUlate the integral numerically starting from the sgap (is it possible?...)
#
elif cost_f_choice == 'L':
    # spectral gap
    axis[0][0].plot(range(spectral_gap.size), spectral_gap, marker='o', color='blue',
                      label='$\delta$', linestyle='-')
    axis[0][0].grid(linestyle='--')
    axis[0][0].set_xlabel('Optimization steps')
    axis[0][0].set_title('Spectral gap $\delta$')
    axis[0][0].set_ylim(0, 1)
    # cost function
    axis[1][0].plot(range(cost_function.size), cost_function, marker='o', color='red',
                      label='Cost f', linestyle='-')
    axis[1][0].grid(linestyle='--')
    axis[1][0].set_xlabel('Optimization steps')
    axis[1][0].set_title('Cost function $' + cost_f_choice + '$')
    # # explored states 
    # exp_states = qmcmc_optimizer.full_explored_states
    # axis[0][1].bar(range(exp_states.size), exp_states, color='green')
    # axis[0][1].grid(linestyle='--')
    # axis[0][1].set_xlabel('Spin configurations')
    # axis[0][1].set_title('Frequency during optimization')
    # # total variational distance 
    # eps = numpy.array(qmcmc_optimizer.db['epsilon'])
    # axis[1][1].plot(range(eps.size), eps, marker='o', color='orange', label='$\epsilon$',
    #                  linestyle='-')
    # axis[1][1].grid(linestyle='--')
    # axis[1][1].set_xlabel('Optimization steps')
    # axis[1][1].set_title('Total variational distance $\epsilon$')
# printing plots
plt.show()
# saving plots
png_name = './simulations_plots/' + cost_f_choice + '/' + core_str + f'_iter_{qmcmc_optimizer.iteration}_' + '.png'
figure.savefig(png_name)
print('\nsaved plot to png file: ' + png_name + '\n')
