
VERSION = 'V4.7'

import numpy
import cmath

class Q_System():
    '''
    QSystem()

    A base class for quantum systems.
    '''

class SpinSystem(Q_System):
    '''
    SpinSystem(n_spins, T, couplings, fields, statevector, decimal, phase)

    Class representing a spin system.

    Initialization parameters
    -------------------------
    n_spins : int
        Number of spins in the system
    T : float
        Temperature
    couplings: numpy.array
        The matrix defining the spins' couplings
    fields: numpy.array
        The array defining the spin-field interactions
    statevector: numpy.array
        The state vector describing the state of the system
    decimal : int
        The decimal number associated with the spin configuration of the system
    phase : float
        The phase of the complex nonzero coefficient in the state vector

    Notes
    -----
    The spin system is described by a state vector and a decimal
    number corresponding to a spin configuration. For example, a system with 3 spins:

    {↓↑↓} -> {010}

    state vector : [0 + 0j, 0 + 0j, A + Bj, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j]
    decimal : 2

    {↓↑↑} -> {011}

    state vector : [0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, A + Bj, 0 + 0j]
    decimal : 6
    
    If a state vector is provided, the system is initialized in the given state vector and the
    decimal number is defined accordingly. Vice versa when a decimal number is provided. 'phase'
    defines 'A' and 'B' when decimal is provided.

    '''

    def __init__(self, n_spins, T, couplings, fields, statevector=None, decimal=None, phase=0):
        self.name = f'{n_spins}_spins_system'
        self.n_spins = n_spins
        self.beta = 1/T 
        self.J = couplings
        self.h = fields
        self.phase = phase

        # if no decimal or statevector is specified, initialize instance with a random spin
        # configuration
        if statevector is not None:
            self.statevector = statevector
        elif decimal is not None:
            self.decimal = decimal
        else:
            self.decimal = numpy.random.choice(2**n_spins)   

    # when updating config, statevector is updated consequently and vice versa
    @property
    def decimal(self):
        return self._decimal
    @decimal.setter
    def decimal(self, new_decimal):
        self._decimal = new_decimal
        # updating statevector
        statevector = numpy.zeros(shape=2**self.n_spins, dtype=complex)
        statevector[new_decimal] += complex(cmath.cos(self.phase), cmath.sin(self.phase))
        self._statevector = statevector
    
    @property
    def statevector(self):
        return self._statevector
    @statevector.setter
    def statevector(self, new_state):
        self._statevector = new_state
        # updating decimal representation
        x = numpy.nonzero(new_state)[0][0]
        self._decimal = int(x)

    def random_state(self):
        '''
        random_state()

        Initializes the system in a random configuration and returns the corresponding state vector.

        Parameters
        ----------

        Returns
        -------
        statevector: numpy.array
            The state vector describing the state of the system
        '''
        self.decimal = numpy.random.choice(2**self.n_spins)
        return self.statevector

    def spins_configuration(self):
        '''
        spins_configuration()

        Returns the spin configuration of the system in the form of an array of +1 (spin up) and -1
        (spin down).

        Parameters
        ----------

        Returns
        -------
        spins_configuration: numpy.array
            Array of +1 (spin up) and -1 (spin down) describing the system's spins configuration
        '''
        # converting decimal number into binary string
        binary_str = bin(int(self.decimal)).replace("0b","")[::-1]
        # creating a spins config from the binary string
        spins_config = numpy.concatenate((
                       numpy.array(list(binary_str), dtype=int), 
                       numpy.zeros(self.n_spins - len(binary_str))
                       ))
        return numpy.array([1 if i==1 else -1 for i in spins_config])

    def energy(self, J_symmetric=True):
        '''
        energy(J_symmetric)

        Returns the energy of the spin system.

        Parameters
        ----------
        J_symmetric : bool
            It speeds up the calculation if the coupling matrix J is symmetric

        Returns
        -------
        energy : float
            Energy of the spin system
        '''
        config = self.spins_configuration()
        if J_symmetric:
            energy = 0.5 * numpy.dot(config.transpose(), -self.J.dot(config)) + \
                    numpy.dot(-self.h.transpose(), config)
        else:
            energy = 0
            for i in range(self.n_spins):
                energy -= self.h[i]*config[i]
                for j in range(i):
                    energy -= self.J[i][j]*config[i]*config[j]
        return energy

    def magnetization(self):
        '''
        magnetization()

        Returns the magnetization of the spin system.

        Parameters
        ----------

        Returns
        -------
        magnetization : float
            Magnetization of the spin system
        '''
        return numpy.sum(self.spins_configuration())