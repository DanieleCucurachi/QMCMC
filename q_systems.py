
VERSION = 'V4.7'

import numpy
import cmath

class Q_System():
    '''
    '''

class SpinSystem(Q_System):
    '''
    da rivedere tutto, bisogna capire se lasciare le cose come metodi oppure se metterle come proprietà
    '''

    def __init__(self, n_spins, T, couplings, fields, statevector=None, decimal=None, phase=0):
        self.name = '{}_spins_system'.format(n_spins)
        self.n_spins = n_spins
        self.beta = 1/T  # 1/kbT
        self.J = couplings
        self.h = fields
        self.phase = phase

        # if no decimal or statevector is specified, initialize instance with a random spin configuration
        if statevector is not None:
            self.statevector = statevector
        elif decimal is not None:
            self.decimal = decimal
        else:
            self.decimal = numpy.random.choice(2**n_spins)   

    # when updating config, statevector is updated consequently and vice versa
    # IO METTREI ANCHE CONFIG E ENERGY COME AUTO UPDATING ATTRIBUTERS
    @property
    def decimal(self):
        return self._decimal
    @decimal.setter
    def decimal(self, new_decimal):
        self._decimal = new_decimal
        # updating statevector
        statevector = numpy.zeros(shape=2**self.n_spins, dtype=complex)   # WARNING, BE CAREFUL YOU DON'T LOSE IMAGINARY PART IN THE CALCULATION
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

    def random_state(self):  #?????
        '''
                    .. parsed-literal::

                     ┌───┐
                a_0: ┤ H ├──■─────────────────
                     └───┘┌─┴─┐
                a_1: ─────┤ X ├──■────────────
                          └───┘┌─┴─┐
                a_2: ──────────┤ X ├──■───────
                               └───┘┌─┴─┐
                b_0: ───────────────┤ X ├──■──
                                    └───┘┌─┴─┐
                b_1: ────────────────────┤ X ├
                                         └───┘

        Parameters:
              a : array_like
              Input array.
              
        Returns:
              tuple_of_arrays : tuple
              Indices of elements that are non-zero.
        '''
        self.decimal = numpy.random.choice(2**self.n_spins)
        return self.statevector

    def spins_configuration(self):
        '''
        '''
        # converting int in binary string
        binary_str = bin(int(self.decimal)).replace("0b","")[::-1]
        # creating a spins config from the binary string
        spins_config = numpy.concatenate((
                       numpy.array(list(binary_str), dtype=int), 
                       numpy.zeros(self.n_spins - len(binary_str))
                       ))
        return numpy.array([1 if i==1 else -1 for i in spins_config])

    def energy(self, J_symmetric=False):
        '''
        qui usa un metodo o l'altro, entrambi non ha senso
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
        qui usa un metodo o l'altro, entrambi non ha senso
        '''
        return numpy.sum(self.spins_configuration())