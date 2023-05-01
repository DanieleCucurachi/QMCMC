
VERSION = 'V4.8'

import numpy
# import mpmath  # for low T 
import random
import pandas
import seaborn as sns
import matplotlib.pyplot as plt

from pandas import DataFrame
from scipy.linalg import expm, kron
from scipy.sparse.linalg import eigs

# defining useful functions
def data_to_collect(optimizer, max_iteration=15e3, delta_cost_f=1):
    '''
    data_to_collect(optimizer, max_iteration, delta_cost_f)

    Establishes whether the algorithms should halt.

    Parameters
    ----------
    max_iteration : int
        Max number of iterations allowed
    delta_cost_f : float
        Minimun change in cost function allowed

    Returns
    -------
    data_to_collect : bool
        False if the optimization has to be stopped
    '''
    return optimizer.iteration <= max_iteration #and optimizer.cost_f_fluctations >= delta_cost_f  #TODO:complete

# defining anstaz classes
class Ansatz():
    '''
    Ansatz(n_spins)

    Initialization parameters
    -------------------------
    n_spins : int
        Number of spins in the system
    '''
    def __init__(self, n_spins):
        self.n_spins = n_spins

    @staticmethod
    def operator(pauli, i, N):
        '''
        operator(pauli, i, N)

        Returns an operator of the right dimension that applies a pauli matrix to the i-th spin in
         a system of N spins.

        Parameters
        ----------
        pauli : numpy.array or numpy.matrix
            The pauli matrix to apply
        i : int
            The target spin's index 
        N : int
            The number of spins in the system
        
        Returns
        -------
        operator : numpy.array
            The pauli operator acting on the i-th spin
        '''
        left = numpy.identity(2**i)
        right = numpy.identity(2 ** (N - i - 1))
        return kron(kron(left, pauli), right)
    
    @staticmethod
    def list_of_op(pauli, N):
        '''
        list_of_op(pauli, N)

        Returns a list of operators of the right dimension that apply a pauli matrix to a single
        spin in a system of N spins. The i-th operator in the list applies a pauli matrix to the 
        i-th spin.

        Parameters
        ----------
        pauli : numpy.array or numpy.matrix
            The pauli matrix to apply
        N : int
            The number of spins in the system

        Returns
        -------
        list_of_op : list of numpy.array
        '''
        return [Ansatz.operator(pauli, i, N) for i in range(N)]
    
    def random_params(self, **kwargs):
        '''
        random_params(self, **kwargs)

        Samples random values of the parameters defining the ansatz. It can use bounds provided when
        alled or default bounds defined in the specific ansatz class.

        Parameters
        ----------
        **kwargs
            The additional keyword arguments

        Returns
        -------
        random_params : numpy.array
        '''
        params = []
        if 'params_bounds' in kwargs.keys():
            params_bounds = kwargs['params_bounds']
        # if no bounds are provided, use the default ones
        else:
            params_bounds = self.params_bounds
        for bounds in params_bounds.values():
            param_value = numpy.random.uniform(low=bounds[0], high=bounds[1], size=None)
            params.append(param_value)
        return numpy.array(params)

class IBM_Ansatz(Ansatz):
    '''
    IBM_Ansatz(n_spins, J, h)

    Class implementing the Quantum-enhaced MCMC anstaz proposed in a recent paper by David Layden et
    al. (https://arxiv.org/abs/2203.12497).

    Initialization parameters
    -------------------------
    n_spins : int
        Number of spins in the system
    J : numpy.array
        The matrix defining the spins' couplings
    h : numpy.array 
        The array defining the spin-field interatctions
    '''
    # defining class attributes
    name = 'IBM'
    
    def __init__(self, n_spins, J, h):
        super().__init__(n_spins)
        self.J = J
        self.h = h 
        self.params_names = ['gamma', 'tau']
        self.params_bounds = {'gamma': (.0, 1), 'tau': (2, 20)}
        self.alpha = numpy.sqrt(self.n_spins) / numpy.sqrt( sum([self.J[i][j]**2 for i in \
                     range(self.n_spins) for j in range(i)]) + sum([self.h[j]**2 for j in \
                                                                    range(self.n_spins)]) )

    def Ising_H(self):
        '''
        Ising_H()

        Returns the Ising hamiltonian defined by the intstance attributes J (couplings) and h (fields)

        Parameters
        ----------

        Returns
        -------
        ham : numpy.array
            Ising hamiltonian defined by couplings J and fields h
        '''
        # generating system size pauli Z operators acting on single spins
        pauli_z = numpy.array([[1, 0], [0, -1]])
        pauli_list = IBM_Ansatz.list_of_op(pauli_z, self.n_spins)
        # generating Ising hamiltonian
        ham = 0
        for i in range(self.n_spins):
            ham -= self.h[i] * pauli_list[i]
            for j in range(i):  
                ham -= self.J[i][j] * pauli_list[i] @ pauli_list[j]  
        return ham

    def unitary(self, params):
        '''
        unitary(params)

        Returns the parametrized unitary implementing the QMCMC ansatz defined in the paper

        Parameters
        ----------
        params : array_like (float)
            The values of the parameters defining the ansatz

        Returns
        -------
        unitary : nd.array
            The unitary implementing the ansatz
        '''
        # parameters defining the ansatz
        gamma = params[0]
        tau = params[1]
        # generating the Ising hamiltonian
        pauli_x = numpy.array([[0, 1], [1, 0]])
        H_prob = self.Ising_H()
        H_mix = sum(IBM_Ansatz.list_of_op(pauli_x, self.n_spins))
        # generating the full Hamiltonian
        H_full = (1-gamma)*self.alpha*H_prob + gamma*H_mix
        return expm(-1j * H_full * tau)   

class Xmix_Ansatz(Ansatz):
    '''
    Xmix_Ansatz(n_spins)

    Class implementing the Xmix ansatz. The Xmix ansatz consist in a sum of pauli X operators acting
    on different spins.

    Initialization parameters
    -------------------------
    n_spins : int
        Number of spins in the system
    *args
        Ignore additional provided parameters
    '''
    # defining class attributes
    name = 'Xmix'
    
    def __init__(self, n_spins, *args):
        super().__init__(n_spins)
        self.params_names = ['tau']
        self.params_bounds = {'tau': (2, 20)}

    def unitary(self, params):
        '''
        unitary(params)

        Returns the parametrized unitary implementing the Xmix ansatz

        Parameters
        ----------
        params : array_like (float)
            The values of the parameters defining the ansatz

        Returns
        -------
        unitary : nd.array
            The unitary implementing the ansatz
        '''
        tau = params[0]
        pauli_x = numpy.array([[0, 1], [1, 0]])
        H_mix = sum(Xmix_Ansatz.list_of_op(pauli_x, self.n_spins))
        H_full = expm(-1j * H_mix * tau) 
        return H_full  

# defining QMCMC classes
class QMCMC_Runner():
    '''
    QMCMC_Runner(spin_system, ansatz)

    Base class implementing the Quantum-enhanced Monte Carlo Markov chain algorithm.

    Initialization parameters
    -------------------------
    spin_system : SpinSystem class instance
        The considered spin system defining the Boltzmann probability we aim to sample from
    ansatz : ansatz class
        The class of the chosen ansatz
    '''
    def __init__(self, spin_system, ansatz):
        self.beta = spin_system.beta
        self.n_spins = spin_system.n_spins
        self.J = spin_system.J
        self.h = spin_system.h
        self.current_state = spin_system.statevector
        self.ansatz = ansatz(self.n_spins, self.J, self.h)
        self.explored_states = numpy.zeros(2**self.n_spins, dtype=int)
    
    def config_from_x(self, x):
        '''
        #TODO: move to SpinSystem() class
        '''
        # converting decimal number into binary string
        binary_str = bin(int(x)).replace("0b","")[::-1]
        # creating a spins config from the binary string
        spins_config = numpy.concatenate((numpy.array(list(binary_str), dtype=int), 
                                          numpy.zeros(self.n_spins - len(binary_str))))
        return numpy.array([1 if i==1 else -1 for i in spins_config])
    
    def config_energy(self, spins_config, J_symmetric=False): 
        '''
        #TODO: move to SpinSystem() class
        '''
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

    def delta(self, i, j, verbose=False):
        '''
        delta(i, j, verbose)

        Calculates the energy difference of two spins configurations.

        Parameters
        ----------
        verbose : bool
            If True, returns the individual spins configurations energies and magnetizations
            If False returns the energy difference of the two spins configurations
        i, j : int, int
            The decimal numbers associated with the spins configurations

        Returns('verbose'==True)
        -------
        prop_state_en, curr_state_en : float, float
            The individual spins configurations energies
        prop_state_mag, curr_state_mag : float, float
            The individual spins configurations magnetizations

        Returns('verbose'==False)
        -------
        delta : float
            Energy difference of the two spins configurations associated with i and j numbers
        '''
        spin_state_i = self.config_from_x(i)
        spin_state_j = self.config_from_x(j)
        prop_state_en = self.config_energy(spin_state_i)
        curr_state_en = self.config_energy(spin_state_j)
        if verbose:
            return prop_state_en, curr_state_en, numpy.sum(spin_state_i), numpy.sum(spin_state_j)
        else:
            return prop_state_en - curr_state_en

    def run_qmcmc_step(self, U):
        '''
        run_qmcmc_step(U)

        Runs a single step of a Quantum-enhanced Monte Carlo Markov chain.

        Parameters
        ----------
        U : numpy.array
            The unitary implementing the quantum proposal strategy

        Returns
        -------
        state_en : float
            Energy of the visited state
        state_mag : float
             Magnetization of the visited state
        '''
        # proposing a new state
        evolved_state = U @ self.current_state
        prob_vector = numpy.array([(abs(a))**2 for a in evolved_state])
        measurement_result = random.choices(range(prob_vector.size), weights=prob_vector, k=1)[0]
        # accepting or rejecting new state
        current_state_idx = numpy.nonzero(self.current_state)[0][0]
        prop_state_en, curr_state_en, prop_state_mag, curr_state_mag = self.delta(measurement_result
                                                                  , current_state_idx, verbose=True)
        A = min(1, numpy.exp(-self.beta * (prop_state_en - curr_state_en)))
        if A >= random.uniform(0, 1):
            # updating current mc state
            self.current_state = numpy.zeros(2**self.n_spins, dtype=complex)
            self.current_state[measurement_result] += (1 + 0j)
            # tracking the visited states during the MC
            self.explored_states[measurement_result] += 1
            # returning observables values
            return prop_state_en, prop_state_mag
        else:
            # tracking the states visited during the MC
            self.explored_states[current_state_idx] += 1
            # returning observables values
            return curr_state_en, curr_state_mag

    def run_random_qmcmc(self, mc_steps, sgap_array, params_bounds={'gamma': (0.2, 0.6), 'tau': (2, 20)}):
        '''
        run_random_qmcmc(mc_steps, sgap_array, params_bounds)

        Runs a Quantum-enhanced Monte Carlo Markov chain sampling random values for the ansatz
        parameters in each step. This is the approach adopted by David Layden et al. in a recent
        paper (https://arxiv.org/abs/2203.12497). The values of the spectral gap at each step are
        stored in an array which, in the end, is appended as a row to an input ndarray. 

        Parameters
        ----------
        mc_steps : int
            The number of Markov chain steps
        sgap_array : numpy.array
            The input array to which the row will be appended 
        params_bounds : dict
            Dictionary containing the intervals from which the parameters defining the ansatz are
            sampled in the format {'parameter name': (lower bound, upper bound), ...}

        Returns
        -------
        sgap_array : numpy.array
            The modified array with the appended row
        '''
        sgap_register = numpy.zeros(mc_steps, dtype=float)
        #
        for step in range(mc_steps):
            # defining random value for the ansatz parameters
            params = self.ansatz.random_params(params_bounds=params_bounds)
            # defining ansatz and calculating respective spectral gap
            sgap_register[step] = self.calculate_sgap(params)
        # saving the results
        sgap_array = numpy.concatenate((sgap_array, sgap_register))
        # sgap_df = sgap_df.append({'sgap mean': sgap_register.mean(), 'sgap std': sgap_register.std()}, ignore_index=True)
        return sgap_array

    def calculate_sgap(self, params):
        '''
        calculate_sgap(params)

        Provided a set parameters defining the quantum proposal distribution, it calculates the
        spectral gap.

        Parameters
        ----------
        params : array_like (float)
            The parameters defining the quantum proposal distribution

        Returns
        -------
        spectral_gap : float
            The value of the spectral gap
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
        # diagonalizing P
        eigenvals = eigs(P, k=2, which='LM', return_eigenvectors=False)
        # returning the spectral gap
        return 1 - abs(eigenvals[-2])  # eigenvals[-1].real - abs(eigenvals[-2])

class QMCMC_Optimizer(QMCMC_Runner):
    '''
    QMCMC_Optimizer(spin_system, ansatz, mc_length, average_over=1, cost_f_choice='ACF',
                   optimization_approach='concatenated_mc', check_point=5000, observable='energy',
                   verbose=True,**kwargs)

    A class implementing the QMCMC optimization algorithm.

    Initialization parameters
    -------------------------
    spin_system : SpinSystem class instance
        The considered spin system defining the Boltzmann probability we aim to sample from
    ansatz : ansatz class
        The class of the chosen ansatz
    mc_length : int
        The number of Markov chain steps to perform
    average_over : int
        # TODO: remove
    cost_f_choice : string
        Specifies the type of cost function used
    optimization_approach : string
        Specifies the type of optimization approach used
    check_point : int
        Defines after how many steps a summary of the current state of the optimization process is
        printed out
    observable : string
        In case 'cost_f_choice'=='ACF', specifies the observable used to calculate the cost function
    verbose : bool
        If True, a summary of the optimization process is printed out at every check point
    **kwargs
        The additional keyword arguments
    '''
    def __init__(self, spin_system, ansatz, mc_length, average_over=1, cost_f_choice='ACF',
                   optimization_approach='concatenated_mc', check_point=5000, observable='energy',
                     verbose=True,**kwargs): 
        super().__init__(spin_system, ansatz)
        self.mc_length = mc_length
        self.average_over = average_over
        self.iteration = 0
        self.check_point = check_point
        self.optimization_approach = optimization_approach
        self.db = DataFrame(columns=['cost f', 'spectral gap'] + self.ansatz.params_names) 
        self.full_explored_states = numpy.zeros(shape=2**self.n_spins, dtype=int)
        self.observable = observable
        self.verbose = verbose
        self.observable_register = None
        self.cost_f_choice = cost_f_choice
        if self.cost_f_choice == 'L':
            self.boltzmann_prob = self.calculate_boltzmann_prob()
        elif self.cost_f_choice == 'ACF':
            self.discard_initial_transient(kwargs['initial_transient'] if 'initial_transient' in kwargs.keys() else self.n_spins * 1e3)
            self.lag = kwargs['lag'] if 'lag' in kwargs.keys() else 5
            if isinstance(self.lag, dict):
                self.cost_f_register = numpy.zeros(3, dtype=float)
                self.current_cost_f_best = numpy.zeros(3, dtype=float)
                self.past_cost_f_best = numpy.zeros(3, dtype=float)
                self.acf_noise = self.lag['acf_noise']  
                self.lag_scale = self.lag['lag_scale']  
                self.lags_array = self.generate_lags_array(self.lag['lag'])
        else:
            raise ValueError('\nProvide valid cost funtion (cost_f_choice):\n- "L"\n- "ACF"\n')

    def discard_initial_transient(self, initial_transient):
        '''
        discard_initial_transient(initial_transient)

        Runs a ceratin number of Markov chain steps without saving the results.

        Parameters
        ----------
        initial_transient : int
            The number of Markov chain steps to run

        Returns
        -------
        '''
        def run_mc(U):
            # proposing a new state
            evolved_state = U @ self.current_state
            prob_vector = numpy.array([(abs(a))**2 for a in evolved_state])
            measurement_result = random.choices(range(prob_vector.size), weights=prob_vector, k=1)[0]
            # accepting or rejecting new state
            current_state_idx = numpy.nonzero(self.current_state)[0][0]
            delta = self.delta(measurement_result, current_state_idx)
            A = min(1, numpy.exp(-self.beta * delta))
            if A >= random.uniform(0, 1):
                self.current_state = numpy.zeros(2**self.n_spins, dtype=complex)
                self.current_state[measurement_result] += (1 + 0j)
        mc_step = 0
        while mc_step < initial_transient:
            gamma = numpy.random.uniform(low=0.2, high=0.6, size=None)  # interval used in IBM's paper
            tau = numpy.random.uniform(low=2, high=20, size=None)  # interval used in IBM's paper
            params = (gamma, tau)
            U = self.ansatz.unitary(params)
            run_mc(U)
            mc_step += 1
        if self.verbose:
            print(f'\n\nDiscarded {int(initial_transient)} samples\n')
    
    def calculate_boltzmann_prob(self):  # spostare in IsingModel()?
        '''
        calculate_boltzmann_prob()

        Calculates the Boltzmann probabilities for the considered spin system.

        Parameters
        ----------

        Returns
        -------
        boltzmann prob : numpy.array
            The array containing the probabilities. The i-th entry represents the Boltzmann prob of
            the configuration whose decimal representation is the number i
        '''
        # calculating Boltzmann probabilities for each spin configuration
        boltzmann_exp = numpy.zeros(shape=2**self.n_spins)
        for i in range(2**self.n_spins):
            boltzmann_exp[i] = (numpy.exp(-self.beta * self.config_energy(self.config_from_x(i))))
        # calculating partition function
        partition_function = sum(boltzmann_exp)
        return numpy.array([i/partition_function for i in boltzmann_exp])

    def delta(self, i, j, verbose=False):  # change verbose name, find better one
        '''
        delta(i, j, verbose)

        Calculates the energy and magnetization difference of two spins configurations.

        Parameters
        ----------
        verbose : bool
            If True, depending on the chosen observable, returns the individual spins
            configurations energies or magnetizations together with the difference
            If False, returns the energy difference of the two spins configurations
        i, j : int, int
            The decimal numbers associated with the spins configurations

        Returns('verbose'==True)
        -------
        prop_state_ , curr_state_ : float, float
            The individual spins configurations energies or magnetizations
        prop_state_mag, curr_state_mag : float, float
            The individual spins configurations magnetizations

        Returns('verbose'==False)
        -------
        delta : float
            Energy difference of the two spins configurations associated with i and j numbers
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
        cost_function()

        Calculates the cost function on a set of Markov chain samples.

        Parameters
        ----------

        Returns
        -------
        cost : float
            The value of the cost function
        '''
        cost = 0
        if self.cost_f_choice == 'L':
            for i in range(2**self.n_spins): 
                for j in range(i):
                    if self.explored_states[i]!=0 and self.explored_states[j]!=0 :
                        numerator = numpy.exp(-self.beta * self.delta(i, j))
                        denominator = self.explored_states[i]/self.explored_states[j]
                        x_i_j = numerator/denominator
                        cost -= (1/((1/x_i_j) + x_i_j))
        elif self.cost_f_choice == 'ACF':
            observable = self.observable_register
            sample_mean = observable.mean()
            # fixed lag
            if isinstance(self.lag, int): 
                for i in range(observable.size - self.lag):
                    cost += (observable[i] - sample_mean)*(observable[i+self.lag] - sample_mean)
                # cost /= (observable.size - self.lag)  # TODO: remove
            # lag optimization
            elif isinstance(self.lag, dict):
                for lag_idx, lag in enumerate(self.lags_array):
                    for i in range(observable.size - lag):
                        self.cost_f_register[lag_idx] +=  \
                                 (observable[i] - sample_mean)*(observable[i+lag] - sample_mean)
                    self.cost_f_register[lag_idx] /= (observable.size - lag) 
                cost = self.cost_f_register[1]
                #
                if cost < self.current_cost_f_best[1] or not self.current_cost_f_best.any():
                    self.current_cost_f_best = self.cost_f_register.copy()
            # lag integration 
            elif self.lag == 'integral':
                lag = 1
                c = 0
                while c >= 0 and lag < observable.size:
                    cost += c
                    c = 0
                    for i in range(observable.size - lag):
                        c += (observable[i] - sample_mean)*(observable[i+lag] - sample_mean)
                    c /= (observable.size - lag)
                    lag += 1
            # exponential decay fit
            # elif self.lag == 'acf_fit':
            #     #TODO: remove

        return cost
    
    def optimize_lag(self):
        '''
        optimize_lag()

        Optimize the lag value during the optimization process.

        Parameters
        ----------

        Returns
        -------
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
        generate_lags_array(lag)

        Provided a lag value, it returns an array containing the original value shifted by 
        -'lag_scale', 0 and +'lag_scale'.

        Parameters
        ----------
        lag : int
            The lag value

        Returns
        -------
        lag_array : numpy.array
            The array containing the three lag values 
        '''
        return numpy.array([lag - self.lag_scale, lag, lag + self.lag_scale])

    def run_qmcmc_step(self, U):
        '''
        run_qmcmc_step(U)

        Runs a single step of a Quantum-enhanced Monte Carlo Markov chain.

        Parameters
        ----------
        U : numpy.array
            The unitary implementing the quantum proposal strategy

        Returns
        -------
        state_obs : float
            The value of the chosen observable of the visited state
        '''
        # proposing a new state
        evolved_state = U @ self.current_state
        prob_vector = numpy.array([(abs(a))**2 for a in evolved_state])
        measurement_result = random.choices(range(prob_vector.size), weights=prob_vector, k=1)[0]
        # accepting or rejecting new state
        current_state_idx = numpy.nonzero(self.current_state)[0][0]
        delta, prop_state_obs, curr_state_obs = self.delta(measurement_result, current_state_idx,
                                                            verbose=True)
        A = min(1, numpy.exp(-self.beta * delta))
        if A >= random.uniform(0, 1):
            self.current_state = numpy.zeros(2**self.n_spins, dtype=complex)
            self.current_state[measurement_result] += (1 + 0j)
            self.explored_states[measurement_result] += 1
            return prop_state_obs
        else:
            self.explored_states[current_state_idx] += 1
            return curr_state_obs

    def __call__(self, params, *args):
        '''
        __call__(params, *args)

        Runs a single step of a Quantum-enhanced Monte Carlo Markov chain.

        Parameters
        ----------
        params : array_like
            The parameters defining the quantum proposal strategy
        *args
            The additional arguments

        Returns
        -------
        cost_f_value : float
            The value of the cost function on a set of Markov chain samples
        '''
        cost_f_values = numpy.zeros(self.average_over, dtype=float) 
        # defining U
        U = self.ansatz.unitary(params)
        # running MCMC proposing states with U(params) 
        if self.optimization_approach == 'concatenated_mc':
            for m in range(self.average_over):
                self.explored_states = numpy.zeros(2**self.n_spins, dtype=int)
                self.observable_register = numpy.zeros(self.mc_length, dtype=float)
                for n in range(self.mc_length):
                    self.observable_register[n] = self.run_qmcmc_step(U)
                # saving current cost function value
                cost_f_values[m] = self.cost_function()
                # updating explored states registers
                self.full_explored_states += self.explored_states

        elif self.optimization_approach == 'same_start_mc':
            for m in range(self.average_over):
                self.explored_states = numpy.zeros(2**self.n_spins, dtype=int)
                self.observable_register = numpy.zeros(self.mc_length, dtype=float)
                self.current_state = args[0]
                for n in range(self.mc_length):
                    self.observable_register[n] = self.run_qmcmc_step(U)
                # saving current cost function realization
                cost_f_values[m] = self.cost_function()
                # updating explored states registers
                self.full_explored_states += self.explored_states
                
        elif self.optimization_approach == 'random_start_mc':
            # starting from a randomly selected state
            init_state = self.spin_system.random_state()
            for m in range(self.average_over):
                self.explored_states = numpy.zeros(2**self.n_spins, dtype=int)
                self.observable_register = numpy.zeros(self.mc_length, dtype=float)
                self.current_state = init_state
                for n in range(self.mc_length):
                    self.observable_register[n] = self.run_qmcmc_step(U)
                # saving current cost function
                cost_f_values[m] = self.cost_function()
                # updating explored states registers
                self.full_explored_states += self.explored_states
        else:
            raise ValueError('Provide valid optimization approach (optimization_approach):' + \
                             '\n- "concatenated_mc"\n- "random_start_mc"\n- "same_start_mc"')

        # counting how many times the cost function is evaluated
        self.iteration += 1 
        if self.iteration % self.check_point == 0 and self.verbose:
            print('\nCost function evaluated:', self.iteration, 'times\n')
        # returning scalar value for scipy to optimize
        return cost_f_values.mean()

    def get_save_results(self, termination_message=False, **kwargs):
        '''
        get_save_results(termination_message, **kwargs)

        Saves the current optimization results in a pandas DataFrame.

        Parameters
        ----------
        termination_message : bool
            The parameters defining the quantum proposal strategy
        **kwargs
            The additional arguments

        Returns
        -------
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
            raise ValueError('Provide valid input:\n- result = scipy OptimizeResult object' + \
                                '\n- params = array like object, cost_f = scalar value')
        # saving current optimization step results
        # self.db = self.db.append(dictionary, ignore_index=True)  # TODO: remove
        self.db = pandas.concat([self.db, DataFrame([dictionary])], axis=0, ignore_index=True) 
        # printing update message
        if self.verbose:
            print(f"\n----------        Current sgap value: {round(dictionary['spectral gap'], 3)},"
                  + f" params values: {params}        ----------\n")

# defining Ising model classes
class IsingModel():
    '''
    IsingModel(n_spins)

    A base class for Ising models.

    Initialization parameters
    -------------------------
    n_spins : int 
        Number of spins in the system
    '''
    def __init__(self, n_spins):
        '''
        '''
        self.n_spins = n_spins
        self.name = 'Ising'

class IsingModel_1D(IsingModel):
    '''
    IsingModel_1D(n_spins, random=True, **kwargs)

    A class for 1D Ising models with periodic (toroidal) boundary conditions.

    Initialization parameters
    -------------------------
    n_spins : int 
        Number of spins in the system
    random : bool
        If True, couplings J and fields h are randomly sampled from a Gaussian prob distribution
    **kwargs
        The additional keyword arguments
    '''
    def __init__(self, n_spins, random=True, **kwargs):
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
        build_J(random, **kwargs)

        Returns the symmetric spin couplings matrix J.

        Parameters
        ----------
        random : bool
            If True, J entries are randomly sampled from a Gaussian prob distribution
        **kwargs
            The additional keyword arguments

        Returns
        -------
        J : numpy.array
            The couplings matrix
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
        # making J symmetric and returning it
        return  numpy.round(J + J.transpose(), decimals=3)

    def build_h(self, random, **kwargs):
        '''
        build_h(random, **kwargs)

        Returns the spin-field interactions vector.

        Parameters
        ----------
        random : bool
            If True, h entries are randomly sampled from a Gaussian prob distribution
        **kwargs
            The additional keyword arguments

        Returns
        -------
        h : numpy.array
            The fields vector
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
        summary(plot)

        Prints out a summary of the model.

        Parameters
        ----------
        plot : bool
            If True, prints out a colormap representing the spin couplings J
        
        Returns
        -------
        '''
        print('\n\n============================================================')
        print('          MODEL : ' + self.name)
        print('============================================================')
        print('Non-zero Interactions (J) : ' + str(int(numpy.count_nonzero(self._J) /2)) + \
                 ' / ' + str(int(0.5 * self.n_spins * (self.n_spins - 1))))
        print('Non-zero Bias (h) : ' + str(int(numpy.count_nonzero(self._h))) + ' / ' + \
               str(self.n_spins))
        print('------------------------------------------------------------')
        print('Average Interaction Strength <|J|>: ', round(numpy.sum(numpy.abs(self._J)) / \
                                                                  numpy.count_nonzero(self._J), 3))
        print('Average Bias Strength <|h|>: ', round(numpy.mean(numpy.abs(self._h)), 3))
        print('------------------------------------------------------------\n\n')

        if plot:
            plt.figure(figsize=(4,3), dpi=100)
            print('Spins coupling heatmap: \n')
            hm = sns.heatmap(self._J, square=True, annot=False, cbar=True)
            hm.set_ylabel('spin index', fontsize=20, labelpad=10)
            hm.set_xlabel('spin index', fontsize=20, labelpad=10)
            hm.tick_params(labelsize=15, axis='both', which='major', pad=10, width=2, length=8)
            hm.figure.axes[1].tick_params(labelsize=15, width=2, length=6)
            hm.collections[0].colorbar.set_label('coupling $J_{ij}$', fontsize=20, labelpad=10)
            plt.show()

class IsingModel_2D(IsingModel_1D):
    '''
    IsingModel_2D(n_spins, random=True, **kwargs)

    A class for 2D Ising models.

    Initialization parameters
    -------------------------
    n_spins : int 
        Number of spins in the system
    random : bool
        If True, couplings J and fields h are randomly sampled from a Gaussian prob distribution
    **kwargs
        The additional keyword arguments
    '''
    def __init__(self, n_spins, random=True, **kwargs):
        super().__init__(n_spins, random, **kwargs)  
        self.name = '2D_' + self.name[3:]

    def build_J(self, random, **kwargs):
        '''
        build_J(random, **kwargs)

        Returns the symmetric spin couplings matrix J.

        Parameters
        ----------
        random : bool
            If True, J entries are randomly sampled from a Gaussian prob distribution
        **kwargs
            The additional keyword arguments

        Returns
        -------
        J : numpy.array
            The couplings matrix
        '''
        def nearest_neigh_coupling(rows, cols, J_value=None, loc=0, scale=1):  
            def generate_J_value(J_value=J_value, loc=loc, scale=scale): 
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
                    nn_J[spin_index][i*cols + (j-1)%cols] += generate_J_value()
                    nn_J[spin_index][((i-1)%rows)*cols + j] += generate_J_value()
            return nn_J
        if 'nearest_neigh' in kwargs.keys() and kwargs['nearest_neigh']: 
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
            J_tril = numpy.tril(J, -1)
            J_triu = J_tril.transpose()
            J = J_tril + J_triu
        else:
            J = J - numpy.diag(numpy.diag(J))
        # return coupling matrix
        return numpy.round(J, decimals=3)
