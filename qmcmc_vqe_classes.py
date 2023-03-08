
VERSION = 'V4.5'

import numpy
import math  # replace math.exp with numpy.exp, cancella math se puoi, be coherent
# import mpmath  # for low T ? better solution?
import random
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
#from copy import deepcopy  # delete
from pandas import DataFrame
from scipy.linalg import expm, kron
from scipy.sparse.linalg import eigs

# TODO: use numpy empty ro initialize arrays that u will fill up
# TODO: check you used copy() when needed to copy array
# potri usare cupy e ray (nell'audio chris spiega come suare + mail a balasz con esempio) 
# ho levato al matrice numpy.matrix
# devia ncora account for low T, il tuo codice si rompe a low T
# from scipy.sparse import identity, kron, csr...  chris says you don't have sparse matrices so there is no advantage
# in using sparse library --> this is because you ahve the exponential U=expm(), think, it makes it not sparse!
# However sparse could be used if you adopot a different time evolution method (carleo)
# import mpmath for low T calculations ####
# #if self.iteration > 2000:  #QUESTO NON VA BENE COSI DEVI TROVARE SOLUZIONE MIGLIORE https://stackoverflow.com/questions/70724216/how-terminate-the-optimization-in-scipy
        #    break

def data_to_collect(optimizer, max_iteration=15e3, delta_cost_f=1e5):
    '''
    '''
    return optimizer.iteration <= max_iteration #and optimizer.cost_f_fluctations >= delta_cost_f  #TODO:complete

class Ansatz():
    '''
    '''
    def __init__(self, n_spins):
        self.n_spins = n_spins

    # HA SENSO USARE STATIC METHOD QUI? 
    @staticmethod
    def operator(pauli, i, N):
        '''
        '''
        left = numpy.identity(2**i)
        right = numpy.identity(2 ** (N - i - 1))
        return kron(kron(left, pauli), right)
    
    @staticmethod
    def list_of_op(pauli, n_spins):
        '''
        '''
        return [Ansatz.operator(pauli, i, n_spins) for i in range(n_spins)]

# defining classes
class IBM_Ansatz(Ansatz):
    '''
    '''
    # defining class attributes
    name = 'IBM'
    
    def __init__(self, n_spins, J, h):
        super().__init__(n_spins)
        self.J = J
        self.h = h 
        self.params_names = ['gamma', 'tau']
        self.alpha = numpy.sqrt(self.n_spins) / numpy.sqrt( sum([self.J[i][j]**2 for i in range(self.n_spins) \
                                for j in range(i)]) + sum([self.h[j]**2 for j in range(self.n_spins)]) )

    def Ising_H(self):
        '''
        if isinstance(n_spins, int):
            pass
        elif isinstance(n_spins, numpy.ndarray):
            n_spins = len(n_spins)
        else:
            raise ('Invalid type. {} is not int or numpy.ndarray'.format(n_spins))
        '''
        # generating system size pauli Z operators acting on single spins
        pauli_z = numpy.array([[1, 0], [0, -1]])
        pauli_list = IBM_Ansatz.list_of_op(pauli_z, self.n_spins)
        # generating Ising hamiltonian (IBM paper)
        ham = 0
        for i in range(self.n_spins):
            ham -= self.h[i] * pauli_list[i]
            for j in range(i):  # implementing new method here (taken from gitlab qmcmc)
                ham -= self.J[i][j] * pauli_list[i] @ pauli_list[j]  
        return ham  # numpy.matrix(ham)
    
    # COL NUOVO METODO TI LIBERI DI CICLI for nested (in realtà no, è solo piu compatto)
    # def calculate_alpha(self, efficient=True, **kwargs):  ## mettere kwargs al posto di H_prob and Hmix
    #     '''
    #     '''
    #     if efficient:
    #         alpha_denominator = 0
    #         for j in range(self.n_spins):
    #             alpha_denominator += self.h[j]**2
    #             for k in range(self.n_spins):
    #                 alpha_denominator += self.J[j][k]**2 if j > k else 0
    #         return math.sqrt(self.n_spins/alpha_denominator)
    #     elif 'H_prob' in kwargs and 'H_mix' in kwargs:
    #         # without using numpy.matrix this doeasn't work
    #         H_prob = kwargs['H_prob']
    #         H_mix = kwargs['H_mix']
    #         H_prob_norm = numpy.trace(H_prob.H @ H_prob)   ### QUA SE NON USO numpy.matrix sul secondo termine della moltiplicazione mi da errore, perchè? leegi sotto
    #         H_mix_norm = numpy.trace(H_mix.H @ H_mix)
    #         return math.sqrt(numpy.sum(H_mix_norm)/numpy.sum(H_prob_norm))  ## qui in pratica viene fuori il fatto che kron aumenta la dimensione dell'oggetto, non crea una matrice piu grande come fai tu carta
    #                                               ## per oviare al fatto che trace da un array (per via di kron somma diagonali su piu dimensioni) sommo sull0array, probabilmente questo non è giusto
    #     else:
    #         raise ValueError('if "efficient" is not True you must provide valid H_prob and H_mix matrices')

    def unitary(self, params):
        '''
        '''
        gamma = params[0]
        tau = params[1]
        pauli_x = numpy.array([[0, 1], [1, 0]])
        H_prob = self.Ising_H()
        H_mix = sum(IBM_Ansatz.list_of_op(pauli_x, self.n_spins))  # numpy.matrix
        H_full = (1-gamma)*self.alpha*H_prob + gamma*H_mix
        return expm(-1j * H_full * tau)   

class Xmix_Ansatz(Ansatz):  ## modifica qmcmc to account for other ansatz (different numbers of params)
    '''
    '''
    # defining class attributes
    name = 'Xmix'
    
    def __init__(self, n_spins, *args):  ## trovare un altro modo per rendere ansatz adattabili (guarda quando definisci l'insatnce in Runner)
        super().__init__(n_spins)
        self.params_names = ['tau']

    def unitary(self, params):
        '''
        '''
        tau = params[0]
        pauli_x = numpy.array([[0, 1], [1, 0]])
        H_mix = sum(Xmix_Ansatz.list_of_op(pauli_x, self.n_spins))  # numpy.matrix
        H_full = expm(-1j * H_mix * tau) 
        return H_full  

class QMCMC_Runner():
    '''
    '''
    def __init__(self, spin_system, ansatz):  #### ansatz=QMCMC_Ansatz, remove kwargs??
        # here u should usse assert to check everything is alright
        self.beta = spin_system.beta
        self.n_spins = spin_system.n_spins
        self.J = spin_system.J
        self.h = spin_system.h
        self.current_state = spin_system.statevector
        self.ansatz = ansatz(self.n_spins, self.J, self.h)
        self.explored_states = numpy.zeros(2**self.n_spins, dtype=int)  # don't like this, find better way
    
    def config_from_x(self, x):  # ???????? DA AGGIORNARE DATO IL NUOVO SPINSYSTEM
        '''
        '''
        # converting int in binary string
        binary_str = bin(int(x)).replace("0b","")[::-1]
        # creating a spins config from the binary string
        spins_config = numpy.concatenate((numpy.array(list(binary_str), dtype=int), numpy.zeros(self.n_spins - len(binary_str))))
        return numpy.array([1 if i==1 else -1 for i in spins_config])
    
    def config_energy(self, spins_config, J_symmetric=False):  # ??????????? DA AGGIORNARE DATO IL NUOVO SPINSYSTEM
        if J_symmetric:
            energy = 0.5 * numpy.dot(spins_config.transpose(), -self.J.dot(spins_config)) + \
                    numpy.dot(-self.h.transpose(), spins_config)
        else:
            energy = 0
            for i in range(spins_config.size):
                energy -= self.h[i]*spins_config[i]
                for j in range(i):
                    energy -= self.J[i][j]*spins_config[i]*spins_config[j]
        return energy

    def delta(self, i, j):
        '''
        '''
        spin_state_i = self.config_from_x(i)
        spin_state_j = self.config_from_x(j)
        prop_state_en = self.config_energy(spin_state_i)
        curr_state_en = self.config_energy(spin_state_j)
        return prop_state_en, curr_state_en, numpy.sum(spin_state_i), numpy.sum(spin_state_j)

    def run_qmcmc_step(self, U):
        '''
        '''
        # proposing a new state
        evolved_state = U @ self.current_state
        prob_vector = numpy.array([(abs(a))**2 for a in evolved_state])
        measurement_result = random.choices(range(prob_vector.size), weights=prob_vector, k=1)[0]
        # accepting or rejecting new state
        current_state_idx = numpy.nonzero(self.current_state)[0][0]
        prop_state_en, curr_state_en, prop_state_mag, curr_state_mag = self.delta(measurement_result, current_state_idx)  # magari inverti in modo da avere lo stesso ordine del paper
        A = min(1, math.exp(-self.beta * (prop_state_en - curr_state_en)))
        if A >= random.uniform(0, 1):
            # updating current mc state
            self.current_state = numpy.zeros(2**self.n_spins, dtype=complex) ## da riscrivere questa, non ha senso cosi, fai tipo una funzione
            self.current_state[measurement_result] += (1 + 0j)
            # tracking the visited states during the MC
            self.explored_states[measurement_result] += 1
            # return observables values
            return prop_state_en, prop_state_mag
        else:
            # tracking the states visited during the MC
            self.explored_states[current_state_idx] += 1
            # return observables values
            return curr_state_en, curr_state_mag

    def observables_convergence_check(self, mc_steps, params):  # DA RIVEDERE DOPO che hai cambiato observables
        '''
        '''
        energy_sum = 0
        magnetization_sum = 0
        mc_step = 0
        sample_mean_energy = numpy.zeros(mc_steps, dtype=float)
        sample_mean_magnetization = numpy.zeros(mc_steps, dtype=float)
        U = self.ansatz.unitary(params)
        pbar = tqdm(total=mc_steps)
        while mc_step < mc_steps:  # non stai includendo l'energia dello stato iniziale
            mc_step_energy, mc_step_magnetization = self.run_qmcmc_step(U)
            energy_sum += mc_step_energy
            magnetization_sum += mc_step_magnetization
            sample_mean_energy[mc_step] = energy_sum/(mc_step+1)  # magari con mean() si puo fare meglio
            sample_mean_magnetization[mc_step] = magnetization_sum/(mc_step+1) 
            mc_step += 1
            pbar.update(1)
        pbar.close()
        return sample_mean_energy, sample_mean_magnetization

    def run_IBM_qmcmc(self, mc_steps, observables_df, sgap_df, run): # DA RIVEDERE DOPO che hai cambiato observables
        '''
        '''
        #TODO: can run only IBM ansatz, maybe it should be universal instead (IN REALTA PENSO FUNZIONI ANCHE CON Xmix)
        # DEVI CAMBIARE self.mc_steps in mc_steps
        energy_register = numpy.zeros(mc_steps, dtype=float)
        magnetization_register = numpy.zeros(mc_steps, dtype=float)
        sgap_register = numpy.zeros(mc_steps, dtype=float)
        for step in range(mc_steps):
            # defining random value for the ansatz parameters QUI PUOI METTERLO COME METHOD IN IBM ANSATZ!
            gamma = numpy.random.uniform(low=0.2, high=0.6, size=None)  # interval used in IBM's paper
            tau = numpy.random.uniform(low=2, high=20, size=None)  # interval used in IBM's paper
            params = (gamma, tau)
            # defining ansatz and calculating respective spectral gap
            U = self.ansatz.unitary(params)
            sgap_register[step] = self.calculate_sgap(params)
            # running MC step and saving the partial results
            energy_register[step], magnetization_register[step] = self.run_qmcmc_step(U)
        # saving the results
        observables_df[f'energy{run}'] = energy_register.tolist()
        observables_df[f'magnetization{run}'] = magnetization_register.tolist()
        sgap_df = sgap_df.append({'spectral gap': sgap_register.mean()})
        return observables_df, sgap_df
        

    def run_fixed_params_qmcmc(self, mc_steps, params, observables_df, run, return_sgap=False): # DA RIVEDERE DOPO che hai cambiato observables
        '''
        '''
        energy_register = numpy.zeros(mc_steps, dtype=float)
        magnetization_register = numpy.zeros(mc_steps, dtype=float)
        # defining ansatz
        U = self.ansatz.unitary(params)
        for step in range(mc_steps):
            # running MC step and saving the partial results
            energy_register[step], magnetization_register[step] = self.run_qmcmc_step(U)
        # saving the results
        observables_df[f'energy{run}'] = self.energy_register.tolist()
        observables_df[f'magnetization{run}'] = self.magnetization.tolist()
        if return_sgap:
            return observables_df, self.calculate_sgap(params)
        else:
            return observables_df

    def calculate_sgap(self, params):  # FIND better solution for autocorrellation time
        '''
        '''
        # calculating U (proposal strategy)
        U = self.ansatz.unitary(params)
        # calculating P (transition matrix)
        P = numpy.zeros(shape=(2**self.n_spins, 2**self.n_spins))
        for i in range(2**self.n_spins):
            for j in range(2**self.n_spins):
                Q_i_j = abs(U[i][j])**2
                A_i_j = min(1, numpy.exp(-self.beta * self.delta(j, i)))
                P[i][j] = A_i_j * Q_i_j
                if i==j:
                    for k in range(2**self.n_spins):
                        P[i][j] += (1 - min(1, numpy.exp(-self.beta * self.delta(k, i)))) * abs(U[i][k])**2
        eigenvals = eigs(P, k=2, which='LM', return_eigenvectors=False)
        return eigenvals[-1].real - eigenvals[-2].real

    # FORSE UNA FUNZIONE COME get_save_dict() ci potrebbe stare
    # def get_save_dict(self, **kwargs):  # maybe can find better solution
    #     '''
    #         results.x: ndarray solution of the optimization
    #         add 'epsilon': self.epsilon
    #     '''
    #     if 'results' in kwargs.keys():
    #         results = kwargs['results']
    #         dictionary = {'cost f': results.fun, 'spectral gap': self.calculate_sgap(results.x)}
    #         for param_idx in range(len(self.ansatz.params_names)):
    #             dictionary[self.ansatz.params_names[param_idx]] = results.x[param_idx]
    #     elif 'params' in kwargs.keys() and 'cost_f' in kwargs.keys():
    #         params = kwargs['params']
    #         dictionary = {'cost f': kwargs['cost_f'], 'spectral gap': self.calculate_sgap(params)}
    #         for param_idx in range(len(self.ansatz.params_names)):
    #             dictionary[self.ansatz.params_names[param_idx]] = params[param_idx]
    #     else:
    #         raise ValueError('Provide valid input:\n- result = scipy OptimizeResult object' + \
    #                             '\n- params = array like object, cost_f = scalar value')
    #     self.db = self.db.append(dictionary, ignore_index=True)

class QMCMC_Optimizer(QMCMC_Runner):
    '''
    '''
    def __init__(self, spin_system, ansatz, mc_length, average_over=1, cost_f_choice='ACF',
                   optimization_approach='concatenated_mc', check_point=5000, observable='energy',
                     verbose=True,**kwargs):  #### ansatz=QMCMC_Ansatz
        # TODO: here u should usse assert to check everything is alright
        super().__init__(spin_system, ansatz)
        self.mc_length = mc_length
        self.average_over = average_over
        self.iteration = 0
        self.check_point = check_point
        self.optimization_approach = optimization_approach
        self.db = DataFrame(columns=['cost f', 'spectral gap'] + self.ansatz.params_names)  #  add 'epsilon',
        self.full_explored_states = numpy.zeros(shape=2**self.n_spins, dtype=int)
        self.observable = observable
        self.verbose = verbose
        self.observable_register = None  # non mi piace
        self.cost_f_choice = cost_f_choice
        if self.cost_f_choice == 'L':
            self.boltzmann_prob = self.calculate_boltzmann_prob()
        elif self.cost_f_choice == 'ACF':
            self.discard_initial_transient(kwargs['initial_transient'] if 'initial_transient' in kwargs.keys() else self.n_spins * 1e3)
            self.lag = kwargs['lag'] if 'lag' in kwargs.keys() else 5
            # TODO: qui gli dai un array di tre dove hai lag, noise e scale, ... trova soluzione migliore, magari usa una dictionary per lag
            if isinstance(self.lag, dict):
                self.cost_f_register = numpy.zeros(3, dtype=float)
                self.current_cost_f_best = numpy.zeros(3, dtype=float)
                self.past_cost_f_best = numpy.zeros(3, dtype=float)
                self.acf_noise = self.lag['acf_noise']  # TODO: use chris suggestion to calculate, penso sia da calcolare dentro qui non da dare da fuori
                self.lag_scale = self.lag['lag_scale']  # TODO: better values? how do we chose it? puoi calcolare la larghezza della curva partendo da tau estimate as 2^kn
                self.lags_array = self.generate_lags_array(self.lag['lag'])
        else:
            raise ValueError('\nProvide valid cost funtion (cost_f_choice):\n- "L"\n- "ACF"\n')

    def discard_initial_transient(self, initial_transient):
        '''
        '''
        def run_mc(U): # TROVA UN ALTRA SOLUZIONE (self NO qui) devi rifare bene sta funzione che runna le mc
            # proposing a new state
            evolved_state = U @ self.current_state
            prob_vector = numpy.array([(abs(a))**2 for a in evolved_state])
            measurement_result = random.choices(range(prob_vector.size), weights=prob_vector, k=1)[0]
            # accepting or rejecting new state
            current_state_idx = numpy.nonzero(self.current_state)[0][0]
            delta = self.delta(measurement_result, current_state_idx)  # magari inverti in modo da avere lo stesso ordine del paper
            A = min(1, math.exp(-self.beta * delta))
            if A >= random.uniform(0, 1):
                self.current_state = numpy.zeros(2**self.n_spins, dtype=complex) ## da riscrivere questa, non ha senso cosi, fai tipo una funzione
                self.current_state[measurement_result] += (1 + 0j)
        #
        mc_step = 0
        while mc_step < initial_transient:
            gamma = numpy.random.uniform(low=0.2, high=0.6, size=None)  # interval used in IBM's paper
            tau = numpy.random.uniform(low=2, high=20, size=None)  # interval used in IBM's paper
            params = (gamma, tau)
            U = self.ansatz.unitary(params)
            run_mc(U)
            mc_step += 1
        #
        if self.verbose:
            print(f'\n\ndiscarded {int(initial_transient)} points\n')
    
    def calculate_boltzmann_prob(self):  # spostare in IsingModel()?
        '''
        '''
        # calculating Boltzmann probabilities for each spin configuration
        boltzmann_exp = numpy.zeros(shape=2**self.n_spins)
        for i in range(2**self.n_spins):
            boltzmann_exp[i] = (numpy.exp(-self.beta * self.config_energy(self.config_from_x(i))))
        # calculating partition function
        partition_function = sum(boltzmann_exp)
        return numpy.array([i/partition_function for i in boltzmann_exp])

    # # l'idea qui sarebbe di staccare calculate_sgap in due parti dove la prima calcola P e la seconda calcola sgap
    # # dato P in input --> così da poter usare P anche per calcolare epsilon. Inoltre la parte che calcola P potrebbe
    # # avrebbe come opzione la possibiloità di dare una U in input, così eviti di doverla calcolare più volte
    # # in run_IBM_qmcmc e run_fixed_params_qmcmc
    # def tot_variational_dist(self):    
    #     normalization = sum(self.explored_states)  # this could be done smarter: n = t+1 (ma devi prima modificare call() in modo che il primo stato sia incluso in quelli esplorati)
    #     mc_pd_approx = numpy.array([i/normalization for i in self.explored_states])
    #     distances = numpy.absolute(self.boltzmann_prob - mc_pd_approx)
    #     return numpy.amax(distances)

    def delta(self, i, j, verbose=False):  # change verbose name, find better one
        '''
        '''
        spin_state_i = self.config_from_x(i)
        spin_state_j = self.config_from_x(j)
        prop_state_en = self.config_energy(spin_state_i)
        curr_state_en = self.config_energy(spin_state_j)
        if verbose:
            if self.observable=='energy':
                return (prop_state_en - curr_state_en), prop_state_en, curr_state_en
            elif self.observable=='magnetization':
                return (prop_state_en - curr_state_en), numpy.sum(spin_state_i), numpy.sum(spin_state_j)
        else:
            return (prop_state_en - curr_state_en)

    def cost_function(self):
        '''
        '''
        cost = 0
        if self.cost_f_choice == 'L':
            for i in range(2**self.n_spins): #### metti len(probabilities vector) e devi mettere un versione che sia ggioran di prob vector
                for j in range(i):
                    if self.explored_states[i]!=0 and self.explored_states[j]!=0 :  # do not count two times the pairs (new method here?)
                        numerator = math.exp(-self.beta * self.delta(i, j))
                        denominator = self.explored_states[i]/self.explored_states[j]
                        x_i_j = numerator/denominator
                        cost -= (1/((1/x_i_j) + x_i_j))
        elif self.cost_f_choice == 'ACF':
            # TODO: use sum([(observable[i] - sample_mean)*(observable[i+lag] - sample_mean) 
            # for i in range(observable.size - lag)])
            observable = self.observable_register
            sample_mean = observable.mean()
            # fixed lag
            if isinstance(self.lag, int): 
                # TODO: WARNING not compatible with 'random_start_mc', find a solution
                for i in range(observable.size - self.lag):
                    cost += (observable[i] - sample_mean)*(observable[i+self.lag] - sample_mean)
                cost /= (observable.size - self.lag)  # TODO: remove?
            # single lag optimization
            elif isinstance(self.lag, dict):  # TODO: usa un dizionario qui
                for lag_idx, lag in enumerate(self.lags_array):
                    for i in range(observable.size - lag):
                        self.cost_f_register[lag_idx] += (observable[i] - sample_mean)*(observable[i+lag] - sample_mean)
                    self.cost_f_register[lag_idx] /= (observable.size - lag)  # do not remove
                cost = self.cost_f_register[1]
                #
                if cost < self.current_cost_f_best[1] or not self.current_cost_f_best.any():
                    self.current_cost_f_best = self.cost_f_register.copy()
            # ACF integral 
            elif self.lag == 'integral':
                lag = 1
                c = 0
                while c >= 0 and lag < observable.size:
                    cost += c
                    c = 0
                    for i in range(observable.size - lag):
                        c += (observable[i] - sample_mean)*(observable[i+lag] - sample_mean)
                    c /= (observable.size - lag)  # TODO: remove?
                    lag += 1
            # TODO: here u can merge these two, the code is identical basically
            # exponential decay fit
            elif self.lag == 'acf_fit':
                acf = []
                lag = 1
                while lag < observable.size:
                    c = 0
                    for i in range(observable.size - lag):
                        c += (observable[i] - sample_mean)*(observable[i+lag] - sample_mean)
                    c /= (observable.size - lag)  # TODO: remove? don't think so
                    if c >= 0:
                        acf.append(c)
                    else:
                        break    
                    lag += 1
                # linearizing the system, and fitting a line to the log of the data (https://stackoverflow.com/questions/3938042/fitting-exponential-decay-with-no-initial-guessing)
                acf = numpy.log(acf)
                # fitting the linearized data
                first_order_coeff, _ = numpy.polyfit(range(1, acf.size + 1), acf, 1)
                # derivating autocorrelation time (tau) to use it as cost function
                cost = - 1/first_order_coeff  # TODO: non mi piace il fatto che dividi di nuovo perdi info anche qua a gratis

            # AGGIUNGERE CHECK SUL LAG MA FORSE MEGLIO FARLO ALL'INIZIO __init__() con assert, così appena crei la classe ti dice
            # che non va bene
        return cost
        # else:
        #     raise ValueError('Provide valid cost funtion (cost_f_choice):\n- "L"\n- "ACF"')  # this is useless as there is already a check for this
    
    def optimize_lag(self):
        '''
        '''
        #
        if not self.past_cost_f_best.any():
            pass
        #
        else:
            visibility = numpy.abs(self.past_cost_f_best - self.current_cost_f_best)  # sostituire con numpy.argmax
            if (visibility[0] - self.acf_noise) > visibility[1] and self.lags_array[0] >= 2:
                self.lags_array = self.generate_lags_array(self.lags_array[0])
            elif (visibility[2] - self.acf_noise) > visibility[1]:
                self.lags_array = self.generate_lags_array(self.lags_array[2])
        # 
        self.past_cost_f_best = self.current_cost_f_best.copy()
        # checking how the lag evolves during the optimization
        if self.verbose:
            print('current lag: ', self.lags_array[1], '\n')
        # resetting the current cost function register for the next optimization step
        self.current_cost_f_best = numpy.zeros(3, dtype=float)

    def generate_lags_array(self, lag):
        '''
        '''
        return numpy.array([lag - self.lag_scale, lag, lag + self.lag_scale])

    # def random_state(self): # bo, non so se ha senso metterlo qui, forse meglio tenerlo in SpinSystem
    #     '''
    #     Inizializing random state
    #     '''
    #     rand_state = numpy.zeros(2**self.n_spins, dtype=complex)  ## da riscrivere questa, non ha senso cosi, fai tipo una funzione
    #     rand_state[random.randint(0, 2**self.n_spins - 1)] += (1 + 0j)  # possibilità di scegliere la phase
    #     return rand_state

    def run_qmcmc_step(self, U):  # QUESTA E' DA RIVEDERE, NON MI PIACE LA SOLUZIONE CHE STAI USANDO
        '''
        '''
        # proposing a new state
        evolved_state = U @ self.current_state
        prob_vector = numpy.array([(abs(a))**2 for a in evolved_state])
        measurement_result = random.choices(range(prob_vector.size), weights=prob_vector, k=1)[0]
        # accepting or rejecting new state
        current_state_idx = numpy.nonzero(self.current_state)[0][0]
        delta, prop_state_obs, curr_state_obs = self.delta(measurement_result, current_state_idx, verbose=True)  # magari inverti in modo da avere lo stesso ordine del paper
        A = min(1, math.exp(-self.beta * delta))
        if A >= random.uniform(0, 1):
            self.current_state = numpy.zeros(2**self.n_spins, dtype=complex) ## da riscrivere questa, non ha senso cosi, fai tipo una funzione
            self.current_state[measurement_result] += (1 + 0j)
            self.explored_states[measurement_result] += 1
            return prop_state_obs
        else:
            self.explored_states[current_state_idx] += 1
            return curr_state_obs

    def __call__(self, params, *args):
        '''
        potrei fare una funzione per runnare le micro mc separata e chiamarla qui
        '''
        # creating arrays to save cost function and total variational distance (epsilon) running values
        cost_f_values = numpy.zeros(self.average_over, dtype=float) # meglio definire prima lo spazio in memoria e poi riempirlo https://stackoverflow.com/questions/568962/how-do-i-create-an-empty-array-and-then-append-to-it-in-numpy
        # defining U
        U = self.ansatz.unitary(params)
        #
        # running MCMC proposing states with U(params) 
        if self.optimization_approach == 'concatenated_mc':  # qua forse leva il fatto che fai la average, non serve con questo approccio
            for m in range(self.average_over):
                self.explored_states = numpy.zeros(2**self.n_spins, dtype=int)
                self.observable_register = numpy.zeros(self.mc_length, dtype=float)
                for n in range(self.mc_length):
                    self.observable_register[n] = self.run_qmcmc_step(U)
                # saving current cost function value
                cost_f_values[m] = self.cost_function() # doesn't work when average_over = 1
                # updating explored states registers (QUANTO FAI LA MEDIA SAREBBE BUONA IDEA PASSARE LA MEDIA DEGLI SATTI ESPLORATI (V2.5))
                self.full_explored_states += self.explored_states

        elif self.optimization_approach == 'same_start_mc':
            for m in range(self.average_over):
                self.explored_states = numpy.zeros(2**self.n_spins, dtype=int)
                self.observable_register = numpy.zeros(self.mc_length, dtype=float)
                self.current_state = args[0]  ## better solution??
                for n in range(self.mc_length):
                    self.observable_register[n] = self.run_qmcmc_step(U)
                # saving current cost function realization
                cost_f_values[m] = self.cost_function()
                # updating explored states registers
                self.full_explored_states += self.explored_states
                
        elif self.optimization_approach == 'random_start_mc':
            #
            init_state = self.spin_system.random_state()
            #
            for m in range(self.average_over):
                self.explored_states = numpy.zeros(2**self.n_spins, dtype=int)
                self.observable_register = numpy.zeros(self.mc_length, dtype=float)
                self.current_state = init_state
                for n in range(self.mc_length):
                    self.observable_register[n] = self.run_qmcmc_step(U)
                # saving current cost function and total variational distance (epsilon) realizations
                cost_f_values[m] = self.cost_function()
                # updating explored states registers
                self.full_explored_states += self.explored_states
        else:
            raise ValueError('Provide valid optimization approach (optimization_approach):' + \
                             '\n- "concatenated_mc"\n- "random_start_mc"\n- "same_start_mc"')

        # counting function evaluations during SciPy optimization (one evaluation corrispond to running m MC of t steps)
        self.iteration += 1  # dovrei metterlo in get_dict_save e contare gli opt steps invece?
        if self.iteration % self.check_point == 0 and self.verbose:
            print('\ncost function evaluated:', self.iteration, 'times\n')
        # returning scalar value for scipy to optimize
        return cost_f_values.mean()

    # def calculate_sgap(self, params, return_ac_time):  # FIND better solution for autocorrellation time
    #     '''
    #     '''
    #     slem = super().calculate_sgap(params)
    #     if return_ac_time:
    #         return slem, -1/(numpy.log(slem))
    #     else:
    #         return slem

    def get_save_results(self, termination_message=False, **kwargs):  # KEEP INCREASING DF SIZE --> BAD PERFORMACES maybe can find better solution
        '''
            results.x: ndarray solution of the optimization
            add 'epsilon': self.epsilon
        '''
        if 'results' in kwargs.keys():
            results = kwargs['results']
            params = results.x
            dictionary = {'cost f': results.fun, 'spectral gap': self.calculate_sgap(params)}
            for param_idx in range(len(self.ansatz.params_names)):
                dictionary[self.ansatz.params_names[param_idx]] = params[param_idx]
            if termination_message:
                print('\noptimization terminated because:', results.message, '\n')
        elif 'params' in kwargs.keys() and 'cost_f' in kwargs.keys():
            params = kwargs['params']
            dictionary = {'cost f': kwargs['cost_f'], 'spectral gap': self.calculate_sgap(params)}
            for param_idx in range(len(self.ansatz.params_names)):
                dictionary[self.ansatz.params_names[param_idx]] = params[param_idx]
        else:
            # here you should use assert
            raise ValueError('Provide valid input:\n- result = scipy OptimizeResult object' + \
                                '\n- params = array like object, cost_f = scalar value')
        # saving current optimization step results
        # self.db = self.db.append(dictionary, ignore_index=True)  # append is deprecated
        self.db = pandas.concat([self.db, DataFrame([dictionary])], axis=0, ignore_index=True) 
        # printing update message
        if self.verbose:
            print(f"\n----------        current sgap value: {round(dictionary['spectral gap'], 3)}, params values: {params}        ----------\n")

class IsingModel():  # FIND BETTER SOLUTION (two classes with inheritance 1DIsing and 2DIsing??)
    '''
    '''
    def __init__(self, n_spins):
        '''
        '''
        self.n_spins = n_spins
        self.name = 'Ising'

class IsingModel_1D(IsingModel):
    '''
    '''
    def __init__(self, n_spins, random=True, **kwargs):  # magari cambia il nome random + kwargs non sono sicyro sia meglio di mettere scale=1
        super().__init__(n_spins)
        self.name = '1D_' + self.name
        self._J = self.build_J(random, **kwargs)
        self._h = self.build_h(random, **kwargs)

    @property
    def J(self):
        return self._J 
    
    @property
    def h(self):
        return self._h

    def build_J(self, random, **kwargs):
        '''
        '''
        if random:
            if 'J_scale' in kwargs.keys():
                scale = kwargs['J_scale']
            else:
                scale = 1
            if 'J_loc' in kwargs.keys():
                loc = kwargs['J_loc']
            else:
                loc = 0
            J_random = numpy.random.normal(loc=loc, scale=scale, size=(self.n_spins))
            J = J_random[:-1]
            J = numpy.diag(J, k=1)
            J[0][self.n_spins-1] = J_random[-1]
            # defining instance name
            self.name += f'_random_mean_{loc}_sd_{scale}'
        elif 'J_value' in kwargs.keys():
            J_value = kwargs['J_value']
            J = numpy.full(self.n_spins-1, J_value)
            J = numpy.diag(J, k=1)
            J[0][self.n_spins-1] = J_value
            # defining instance name
            self.name += f'_Jval_{round(J_value, 2)}'
        else:
            J_value = numpy.random.uniform(low= -2, high= 2, size=None)
            J = numpy.full(self.n_spins-1, J_value)
            J = numpy.diag(J, k=1)
            J[0][self.n_spins-1] = J_value
            # defining instance name
            self.name += f'_Jval_{round(J_value, 2)}'
        # making J symmetric and return couplings matrix
        return  numpy.round(J + J.transpose(), decimals=3)

    def build_h(self, random, **kwargs):
        '''
        '''
        if random:
            if 'h_scale' in kwargs.keys():
                scale = kwargs['h_scale']
            else:
                scale = 1
            if 'h_loc' in kwargs.keys():
                loc = kwargs['h_loc']
            else:
                loc = 0
            h = numpy.random.normal(loc=loc, scale=scale, size=(self.n_spins))
        elif 'h_value' in kwargs.keys():
            h_value = kwargs['h_value']
            h = numpy.full(self.n_spins, h_value)
            # defining instance name
            self.name += f'_hval_{round(h_value, 2)}'
        else:
            h_value = numpy.random.uniform(low=-2, high=2, size=None)
            h = numpy.full(self.n_spins, h_value)
            # defining instance name
            self.name += f'_hval_{round(h_value, 2)}'
        # return fields vector
        return numpy.round(h, decimals=3)
    
    def summary(self, plot=True):
        '''
        '''
        print('\n\n============================================================')
        print('          MODEL : ' + self.name)
        print('============================================================')
        print('Non-zero Interactions (J) : ' + str(int(numpy.count_nonzero(self._J) /2)) + \
                 ' / ' + str(int(0.5 * self.n_spins * (self.n_spins - 1))))
        print('Non-zero Bias (h) : ' + str(int(numpy.count_nonzero(self._h))) + ' / ' + str(self.n_spins))
        print('------------------------------------------------------------')
        print('Average Interaction Strength <|J|>: ', round(numpy.sum(numpy.abs(self._J))/numpy.count_nonzero(self._J), 3))
        print('Average Bias Strength <|h|>: ', round(numpy.mean(numpy.abs(self._h)), 3))
        print('------------------------------------------------------------\n\n')

        if plot:
            plt.figure()  # figsize=(16,10)
            print('Spins coupling heatmap: \n')
            sns.heatmap(self._J, square=True, annot=False, cbar=True).set(xlabel='Spin index', ylabel='Spin index')
            plt.show()

class IsingModel_2D(IsingModel_1D):
    '''
    '''
    def __init__(self, n_spins, random=True, **kwargs):  # magari cambia il nome random + kwargs non sono sicyro sia meglio di mettere scale=1
        super().__init__(n_spins, random, **kwargs)  # rimettre nearest_neigh as a keyword outside kwargs
        self.name = '2D_' + self.name[3:]
        # self._J = self.build_J(random, nearest_neigh, **kwargs)

    # @property
    # def J(self):
    #     return self._J 

    def build_J(self, random, **kwargs):
        '''
        '''
        def nearest_neigh_coupling(rows, cols, J_value=None, loc=0, scale=1):  # better solution? should I put self in here? NO
            def generate_J_value(J_value=J_value, loc=loc, scale=scale):  # bette solution
                if J_value is None:
                    return numpy.random.normal(loc=loc, scale=scale, size=None)
                else:
                    return J_value
            nn_J = numpy.zeros((self.n_spins, self.n_spins))
            for i in range(rows):
                for j in range(cols):
                    spin_index = i*cols + j
                    nn_J[spin_index][i*cols + (j+1)%cols] += generate_J_value() if cols>2 else 0
                    nn_J[spin_index][((i+1)%rows)*cols + j] += generate_J_value() if rows>2 else 0
                    nn_J[spin_index][i*cols + (j-1)%cols] += generate_J_value() # (0-h)%n = n-h
                    nn_J[spin_index][((i-1)%rows)*cols + j] += generate_J_value()
            return nn_J
        if 'nearest_neigh' in kwargs.keys() and kwargs['nearest_neigh']:  # already symmetric and diagonal=0
            # defining spins grid structure
            if 'spins_grid' in kwargs.keys():
                rows, cols = kwargs['spins_grid']
            else:
                raise ValueError('You must provide a valid spin grid')
                #TODO: what if spins grid is not provided? 
            if random:
                if 'J_scale' in kwargs.keys():
                    scale = kwargs['J_scale']
                else:
                    scale = 1
                if 'J_loc' in kwargs.keys():
                    loc = kwargs['J_loc']
                else:
                    loc = 0
                J = nearest_neigh_coupling(rows, cols, J_value=None, loc=loc, scale=scale)
                # defining instance name
                self.name += f'_random_mean_{loc}_sd_{scale}'
            elif 'J_value' in kwargs.keys():
                J_value = kwargs['J_value']
                J = nearest_neigh_coupling(rows, cols, J_value)
                # defining instance name
                self.name += f'_Jval_{round(J_value, 2)}'
            else:
                J_value = numpy.random.uniform(low=-2, high=2, size=None)
                J = nearest_neigh_coupling(rows, cols, J_value)
                # defining instance name
                self.name += f'_Jval_{round(J_value, 2)}'
        else:
            if random:
                if 'J_scale' in kwargs.keys():
                    scale = kwargs['J_scale']
                else:
                    scale = 1
                if 'J_loc' in kwargs.keys():
                    loc = kwargs['J_loc']
                else:
                    loc = 0
                J = numpy.random.normal(loc=loc, scale=scale, size=(self.n_spins, self.n_spins))
                # defining instance name
                self.name += f'_random_mean_{loc}_sd_{scale}'
            elif 'J_value' in kwargs.keys():
                J_value = kwargs['J_value']
                J = numpy.full((self.n_spins, self.n_spins), J_value)
                # defining instance name
                self.name += f'_Jval_{round(J_value, 2)}'
            else:
                J_value = numpy.random.uniform(low=-2, high=2, size=None)
                J = numpy.full((self.n_spins, self.n_spins), J_value)
                # defining instance name
                self.name += f'_Jval_{round(J_value, 2)}'
        # making J symmetric
        if random:
            # J =  (J + J.transpose())/numpy.sqrt(2)  # 1/sqrt(2) so that J ~ N(0, Var(J)) after the symmetrization
            # J = numpy.round(J - numpy.diag(numpy.diag(J)) , decimals=3)
            J_tril = numpy.tril(J, -1)
            J_triu = J_tril.transpose()
            J = J_tril + J_triu
        else:
            J = J - numpy.diag(numpy.diag(J))
        # return coupling matrix
        return numpy.round(J, decimals=3)
