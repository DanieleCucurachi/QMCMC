import numpy as np
import random
from scipy.linalg import expm
from scipy.optimize import minimize
from pandas import DataFrame
# from sklearn.preprocessing import normalize
import seaborn as sns
import matplotlib.pyplot as plt
import os

'''A = np.random.rand(3, 3)
B = np.random.rand(3, 3)
v = np.random.rand(3)

t = 1
delta = 1j * t

U = expm(-delta * B)

print(B)
print(v)
print(U @ v)

for i in range(3):
        for j in range(3):
            if i>j:
                print(i, j)

a = np.array([1 + 1j, 2 + 2j])
b = np.array([abs(f)**2 for f in a])
print(b)

def decimalToBinary(n):
    return bin(n).replace("0b","")[::-1]

print(decimalToBinary(8))
print(type(decimalToBinary(8)))

string ='01001'
print(numpy.array(list(string), dtype=int))

def config_from_x(x, system_size):  #  TODO: BETTER NAME FOR VARIABLES AND FUNCTION AND MAYBE NO STATIC METHOD HERE, OR A DIFFERENT CLASS JUST FOR CALCULATIONS (system_size is basically n_spins, config is spins_config)
        # converting int in binary string
        binary_str = bin(x).replace("0b","")[::-1]
        # creating a spins config from the binary string
        config = numpy.concatenate((numpy.array(list(binary_str), dtype=int), numpy.zeros(system_size - len(binary_str))))
        return numpy.array([1 if config_ele else -1 for config_ele in config])

print(config_from_x(8, 10))

def mat(a):
    return numpy.array([[0, a],[1, a-1]])
a = [mat(b) for b in range(3)]
print(sum(a))

a = numpy.array([[1, 2], [3, 4]])
b = numpy.array([[1, 2], [3, 4]])
print(numpy.matrix(a).H @ numpy.matrix(b))

a = numpy.array([1,2,3,4,5])
b = numpy.array([10, 10, 10, 10, 60])

measurement_result = random.choices(a, weights=b, k=1)
print(measurement_result)

arr = numpy.array([0, 0, 0, 1j, 0, 0, 0])
arra = numpy.array([0, 0, 0, 1j, 0, 1, 0])
print(numpy.nonzero(arr)[0])
print(numpy.nonzero(arra))

print(arr[numpy.nonzero(arr)[0]])

def config_from_x(x, n_spins):  # TODO: WHERE TO PUT THIS? CHANGE NAME 
    # converting int in binary string
    binary_str = bin(x).replace("0b","")[::-1]
    # creating a spins config from the binary string
    spins_config = numpy.concatenate((numpy.array(list(binary_str), dtype=int), numpy.zeros(n_spins - len(binary_str))))
    return numpy.array([1 if i==1 else -1 for i in spins_config])

print(config_from_x(0, 5))

# funzioni vedono anche le variabili fuori, ma da fuori non vedi le variabili definite dentro
b = 2
def abab(a):
    return a+b

print(abab(2))

# how to properly use pandas dataframes
columns = ['spin_state', 'cost_f', 'gamma', 't', 'spectral_gap']
db = DataFrame(columns=columns) 
db = db.append({'spin_state': 1, 'cost_f': 2, 'gamma': 3, 't': 4, 'spectral_gap': 5}, ignore_index=True)

print(db)

# 
prob_vector = [0.2, 0.3, 0.4, 0.1]
measurement_result = random.choices(range(len(prob_vector)), weights=prob_vector, k=1)
print(measurement_result)


# how to work with complex numbers
I = numpy.identity(2)
z = numpy.array([[1, 0], [0, -1]])
a = numpy.array([1+1j, 2+2j])
print(z @ I @ a)

class salsa:
    def __init__(self, a, b, c):
        self.a = a
        if True:
            self.b=b
        else:
            self.b = 0
        self.c = self.def_c(c)

    def def_c(self, c):
        return c+10

maio = salsa(1,2,3)

print(maio.b, maio.c)

# updating an attribute when updating another one
class salsa:
    def __init__(self, a):
        self.a = a

    @property
    def a(self):
        return self._a
    @a.setter
    def a(self, val):
        self._a = val
        self._bb = self._a+10

    @property
    def bb(self):
        return self._bb
    @bb.setter
    def bb(self, val):
        self.a = val-10

maio=salsa(1)
print(maio.a, '\n\n', maio.bb, '\n\n')
maio.a = 3
print(maio.a, '\n\n', maio.bb, '\n\n')
maio.bb = 15
print(maio.a, '\n\n', maio.bb, '\n\n')


from src.q_systems import SpinSystem

prova = SpinSystem(2, 1, 1, 1, config=[1, 1])

print('\n\n')
print(prova.statevector)
prova.config= numpy.array([-1,-1])
print(prova.statevector)
print(prova.config)
prova.statevector = numpy.array([0, 1+0j, 0, 0])
print(prova.config)

print('\n\n', prova.decimal_rep())
prova.config= [-1,-1]
print('\n\n', prova.decimal_rep())




from src.q_systems import SpinSystem

prova = SpinSystem(2, 1, 1, 1, config=[-1, 1])

print('\n\n')
print(prova.statevector)
print(prova.config)

prova.statevector = numpy.array([0, 1+0j, 0, 0])
print(prova.statevector)
print(prova.config)

print('\n\n')
print('\n\n')

prova.config= numpy.array([1,-1])
print(prova.statevector)
print(prova.config)

prova.statevector = numpy.array([0, 0, 1+0j, 0])
print(prova.statevector)
print(prova.config)

####

#funzionamento listea = list('uno')
a.append('due')
print(a)

# scipy optimize
from scipy.optimize import minimize

class alfa():

    def __init__(self, a, b):
        self.a = a
        self.iter = 0
        self.b = b
        self.list = []

    def calcola(self, gamma):
        self.a = 2
        cost = self.a*gamma**2
        self.iter += 1
        self.list.append('uno')
        return cost

inst = alfa(1,2)
param = 10

while inst.iter < 50:
    if inst.iter % 10 ==0:
        print('iter:', inst.iter)
    result = minimize(inst.calcola, param)
    param = result.x

print(result.x)
print(inst.list)
print(inst.iter)

# TEST FUNZIONI 
def config_from_x(x, n_spins):  #TODO: WHERE TO PUT THIS? CHANGE NAME

    # converting int in binary string

    binary_str = bin(int(x)).replace("0b","")[::-1]

    # creating a spins config from the binary string

    spins_config = numpy.concatenate((numpy.array(list(binary_str), dtype=int), numpy.zeros(n_spins - len(binary_str))))

    return numpy.array([1 if i==1 else -1 for i in spins_config])


def config_energy(spins_config,h, J):  #TODO: WHERE TO PUT THIS? CHANGE NAME
    energy = 0
    for i in range(len(spins_config)):
        energy -= h[i]*spins_config[i]
        for j in range(len(spins_config)):
            energy -= J[i][j]*spins_config[i]*spins_config[j] if i > j else 0
    return energy  

print(config_from_x(63, 6))
print(config_energy([-1,-1,-1,-1], [1,1,1,1], [[1,1,1,1], [1,1,1,1], [1,1,1,1], [1,1,1,1]]))

for i in range(4):
    for j in range(4):
        if i > j:
            print(i, j)


from src.q_systems import SpinSystem

n_spins = 6
beta = 1/10
J = numpy.random.normal(loc=0.0, scale=1.0, size=(n_spins, n_spins))  # normal distributed as in the paper
h = numpy.random.normal(loc=0.0, scale=1.0, size=n_spins)
spin_system = SpinSystem(n_spins, beta, J, h)

print(spin_system.config)

# saving pandas dataframes
gamma = 0.25
tau = 2
t = 300
opt = 'cacca'
columns = ['gamma', 'tau', 't']
db = DataFrame(columns=columns) 
db.append({'gamma': gamma, 'tau': tau, 't':t}, ignore_index=True)

db.to_csv('./qmcmc_drafts/simulations_results/gamma' + opt + f'_{gamma}_tau_{tau}_t_{t}.csv', encoding='utf-8')


# normalize array like data with sklearn
a = [1,2,3,4,5,6,7,8,9,10]
b = numpy.array([1,2,3,4,5,6,7,8,9,10])
norm = [float(i)/sum(a) for i in a]
print(norm)
a = numpy.array(norm)
print(b-a)



# 3d plotting

def f(x, y):
    return x+y

size = 3

x = np.linspace(-1, 1, size)
y = np.linspace(-1, 1, size)

Z = np.zeros(shape=(size, size), dtype=float)

for i, xx in enumerate(x):
    for j, yy in enumerate(y):
        Z[j][i] = f(xx, yy) 

X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()



# numpy array append

a = np.array()

a.append(1)
a.append(2)

print(a)


###
class alfa:
    def __init__(self, a, b=1):
        self.a = a
        self.b = b

aaa = alfa(3)
bbb = alfa(3, 2)

print('aaa:', aaa.b, 'bbb:', bbb.b)


# 

lista = [1,2,3,4,5]
numpy_list = np.array(lista)

print(lista[1:3])
print(numpy_list[1:3])


##
apha = 0
for i in range(3):
    for j in range(3):
        apha += i*j if i > j else 0
print(apha)

## OLD SPINSYSTEM CLASS

class SpinSystem(QSystem):

    def __init__(self, n_spins, T, couplings, fields, config=None, phase=0):
        self.name = '{}_spins_system'.format(n_spins)
        self.n_spins = n_spins
        self.beta = 1/T  # 1/kbT
        self.J = couplings
        self.h = fields
        self.phase = phase

        # if no spins configuration is specified, initialize instance with a random one
        if config is None:  # TODO: FIND BETTER WAY TO DO THIS
            self.config = numpy.random.choice([1,-1], size=n_spins)
        else:
            self.config = config

    # when updating config, statevector is updated consequently and vice versa
    @property
    def config(self):
        return self._config
    @config.setter
    def config(self, new_config):
        self._config = new_config
        x = 0
        for i in range(len(self._config)):
            x += (2**i) if self._config[i]==1 else 0
        statevector = numpy.zeros(shape=2**self.n_spins, dtype=complex)   # WARNING, BE CAREFUL YOU DON'T LOSE IMAGINARY PART IN THE CALCULATION
        statevector[x] += complex(cmath.cos(self.phase), cmath.sin(self.phase))
        self._statevector = statevector
    
    @property
    def statevector(self):
        return self._statevector
    @statevector.setter
    def statevector(self, new_state):
        x = numpy.nonzero(new_state)[0]
        # converting int in binary string
        binary_str = bin(int(x)).replace("0b","")[::-1]
        # creating a spins config from the binary string
        spins_config = numpy.concatenate((numpy.array(list(binary_str), dtype=int), numpy.zeros(self.n_spins - len(binary_str))))
        self.config = numpy.array([1 if i==1 else -1 for i in spins_config])

    def energy(self):
        energy = 0
        for i in range(self.n_spins):
            energy -= self.h[i]*self.config[i]
            for j in range(self.n_spins):
                energy -= self.J[i][j]*self.config[i]*self.config[j] if i > j else 0
        return energy

    def decimal_rep(self):
        x = 0
        for i in range(self.n_spins):
            x += (2**i) if self.config[i]==1 else 0
        return x

## test new statevector thing spinsystem
from src.q_systems import SpinSystem

a = SpinSystem

def config_energy(spins_config,h, J):  #TODO: WHERE TO PUT THIS? CHANGE NAME
    energy = 0
    for i in range(len(spins_config)):
        energy -= h[i]*spins_config[i]
        for j in range(len(spins_config)):
            energy -= J[i][j]*spins_config[i]*spins_config[j] if i > j else 0
    return energy

J = np.random.normal(loc=0.0, scale=1.0, size=(3, 3))  # normal distributed as in the paper
h = np.random.normal(loc=0.0, scale=1.0, size=3)
spinsystem = a(3, 10, J, h,statevector=np.array([0,0,0,0,0,0,1,0]))
print(spinsystem.config)
#spinsystem = SpinSystem(3, 10, J, h, config=np.array([-1,  -1, -1]))
#print(spinsystem.statevector)

print(spinsystem.energy(), '\n\n', config_energy(spinsystem.config,h,J))

# ATTENTOOO
print(spinsystem.energy)
print(spinsystem.decimal_rep())



##

def funzione(a,b,**kwargs):
    print(a, b)
    if kwargs['cane'] is not None:
        print(kwargs['cane'])

funzione(1,2, cane='bau', gatto='miao')

# optimize minimize with args and kwargs

def objective(speed, *args, c=12):
    if len(args) > 0:
        summa = speed**2 - c 
        for arg in args:
            summa += arg
        return summa 
    else:
        return speed**2 - c


result = minimize(objective, 3, args=(4,4), method='nelder-mead')
print(result.fun)

pauli_z = np.array([[1, 0], [0, -1]])
U = expm(-1j * pauli_z)
a = np.array([1,1])

print(U @ a)


for i in range(4):
    for j in range(4):
        print(i, j)

        

a = np.array([1,1])
b = np.array([1,4])
for i in range (2):
    a += b
print(a)


from pandas import concat

db = DataFrame(columns=['acca', 'racca'])
for i in range(5):
    dictionary = {'acca': i, 'racca': i+1}
    db = db.append(dictionary, ignore_index=True)

db2 = DataFrame(columns=['acca'])
db2 = db2.append({'acca': 55}, ignore_index=True)

db_f = concat([db2, db]).reset_index(drop=True)

print(db_f)


# scipy optimize
from scipy.optimize import minimize

def calcola(gamma):
        cost = gamma**2 + 2
        return cost

param = 10

result = minimize(calcola, param)

print(result.x, result.fun, result.message)

# dividere un int array ti da indietro float come vorresti
a = np.zeros(3, dtype=int)
b = np.array([1, 3, 8])

for i in range(len(a)):
    a[i] += 1

print(a/b)


params_dict = {'gamma': 0.1, 'tau': 1}
params_guess = np.fromiter(params_dict.values(), dtype=float) 
print(params_guess)


## test new statevector thing spinsystem
from src.q_systems import SpinSystem

a = SpinSystem

def config_energy(spins_config,h, J):  #TODO: WHERE TO PUT THIS? CHANGE NAME
    energy = 0
    for i in range(len(spins_config)):
        energy -= h[i]*spins_config[i]
        for j in range(len(spins_config)):
            energy -= J[i][j]*spins_config[i]*spins_config[j] if i > j else 0
    return energy

J = np.random.normal(loc=0.0, scale=1.0, size=(3, 3))  # normal distributed as in the paper
h = np.random.normal(loc=0.0, scale=1.0, size=3)
spinsystem = a(3, 10, J, h,statevector=np.array([0,0,1+0j,0,0,0,0,0], dtype=complex))
print()
print(spinsystem.spins_configuration())
print(spinsystem.decimal)
print()
b = SpinSystem(3, 10, J, h, decimal=4)
print(b.statevector)
print(b.spins_configuration())
print()
c = SpinSystem(3, 10, J, h)
print(c.statevector)
print(c.spins_configuration())
print(c.decimal)
print()
print('energy from class:', b.energy(), config_energy(b.spins_configuration(), h, J))
c.statevector=np.array([0,0,1+0j,0,0,0,0,0], dtype=complex)
print()
print(c.statevector)
print(c.spins_configuration())
print(c.decimal)
print()
c.decimal=6
print()
print(c.statevector)
print(c.spins_configuration())
print(c.decimal)
print()

# ATTENTOOO


## test new statevector thing spinsystem
from src.q_systems import SpinSystem

a = SpinSystem
n_spins=3

def config_energy(spins_config,h, J):  #TODO: WHERE TO PUT THIS? CHANGE NAME
    energy = 0
    for i in range(len(spins_config)):
        energy -= h[i]*spins_config[i]
        for j in range(len(spins_config)):
            energy -= J[i][j]*spins_config[i]*spins_config[j] if i > j else 0
    return energy

J = np.full((n_spins, n_spins), 1)  # normal distributed as in the paper
h = np.full(n_spins, 1)
system = a(n_spins, 10, J, h,statevector=np.array([0,0,1+0j,0,0,0,0,0], dtype=complex))
print(system.decimal)
print(system.statevector)
system.random_state()
print()
print(system.decimal)
print(system.statevector)
system.random_state()
print()
print(system.decimal)
print(system.statevector)


##
n_spins = 3
J = np.full((n_spins, n_spins), 1)
h = np.full(n_spins, 1)

print(h, '\n\n', J)


a = 10e3

def func(b: int) -> float:
    print(b+0.5)

func(4.123)


# mena numpy array over different axis

a = np.array([[1,1,1],
              [5,5,5],
              [3,3,3]])
b = a.mean(axis=1)

print('\n\nmean axis=0:', a.mean(axis=0))
print('\n\nmean axis=1:', b, 'type:', type(b))
print('\n\n', a[0, 2])  # [riga, colonna]


# rounding element sin numpy arrays

a = np.array([[12,15,46],
              [54,45,56],
              [36,32,36]])

print(np.rint(a.mean(axis=0)), type(np.rint(a.mean(axis=0))))

# riga colonna creazioen arrays numpy
print(np.zeros((3, 6)))


###

# mena numpy array over different axis

a = np.array([[1,1,1],
              [2,2,2],
              [1,2,3]])
b = a.mean(axis=1, dtype=int)

print('\n\nmean axis=0:', a.mean(axis=0, dtype=int))
print('\n\nmean axis=1:', b, 'type:', type(b))
print('\n\n', a[0, 2])  # [riga, colonna]


# nonzero numpy

import numpy

arr = numpy.array([0, 0, 0, 1j, 0, 0, 0])
arra = numpy.array([0, 0, 0, 1j, 0, 1, 0])
print('prima:', numpy.nonzero(arr)[0], 'type:', type(numpy.nonzero(arr)[0]))
print('prima:', numpy.nonzero(arr)[0][0], 'type:', type(numpy.nonzero(arr)[0][0]))
print('seconda:', numpy.nonzero(arra)[0][0], 'type:', type(numpy.nonzero(arr)[0][0]))

print('vediamos e funziona da indice', arr[numpy.nonzero(arr)[0]], arr[numpy.nonzero(arr)[0][0]])



array = np.zeros(shape=(1))
print(array.mean())
a = {'gaga': 1, 'bebe': 2}
print(a.keys())
print('Provide valid optimization_approach:\n    - "concatenated_mc"\n    - "random_start_mc"\n    - "same_start_mc"')


from scipy.linalg import expm, eigh, kron

P = np.array([[0.1, 0.4, 0.5],
              [0.1, 0.4, 0.5],
              [0.1, 0.4, 0.5]])

eigenvals = eigh(P, eigvals_only=True)

print(eigenvals)


n = np.array([1,2,3,4,5,6,7,8,9,10])

print(n[5:])



from tqdm import tqdm

pbar = tqdm(total=10)
mc_steps = 0
a = np.zeros((1000,1000))
while mc_steps < 10:  # non stai includendo l'energia dello stato iniziale
    mc_steps += 1
    for i in range(1000):
        for j in range(1000):
            a[i][j] +=1
    pbar.update(1)
pbar.close()

# ALPHA CALCULATION

# stringa = [f'{i}{j}' for i in range(3) for j in range(i)]
# print(stringa)
stringa = []
for j in range(3):
    for k in range(3):
        if j > k: 
            stringa.append(f'{j}{k}')
print(stringa)

# stringa = [f'{i}{j}' for i in range(3) for j in range(i)]
# print(stringa)
stringa = []
for j in range(3):
    for k in range(j):
            stringa.append(f'{j}{k}')
print(stringa)


# ISING MODEL 2D COUPLING

JJ = 1  # coupling
N = 9  # number of spins
n = int(np.sqrt(N))
J = np.zeros((N,N))
spin_col_ind, spin_row_ind = np.meshgrid(range(n), range(n), indexing='ij')
print(spin_col_ind, '\n\n', spin_row_ind)

for i in range(n):
    for j in range(n):
        spin_ind = i*n + j
        J[spin_ind][i*n + (j+1)%n] += JJ/2
        J[spin_ind][((i+1)%n)*n + j] += JJ/2
        J[spin_ind][i*n + (j-1)%n] += JJ/2  # (0-h)%n == n-h
        J[spin_ind][((i-1)%n)*n + j] += JJ/2

print(J)
nzx, nzy = np.nonzero(J)

for l in range(len(nzx)):
    print(nzx[l], nzy[l])


# NEW ENEGRY CALCULATION METHOD

from src.q_systems import SpinSystem
import time
start_time = time.time()

n_spins = 25
T=10
J = np.random.normal(loc=0.0, scale=1.0, size=(n_spins, n_spins))  # normal distributed as in the IBM paper
J = 1/np.sqrt(2) * (J + J.transpose() )
J = np.round( J - np.diag(np.diag(J)) , decimals= 3)
h = np.round(np.random.normal(loc=0.0, scale=1.0, size=n_spins), decimals=3)

spin_system = SpinSystem(n_spins, T, J, h)
spins_config = spin_system.spins_configuration()

def config_energy(J, h, spins_config):  # ??????????? DA AGGIORNARE DATO IL NUOVO SPINSYSTEM
    energy = 0
    for i in range(len(spins_config)):
        energy -= h[i]*spins_config[i]
        for j in range(i):
            energy -= J[i][j]*spins_config[i]*spins_config[j]
    return energy  
init = time.time()

print('\n\n')
print("--- %s start ---" % (time.time() - start_time))
print('\n\n')
print('my method:', config_energy(J, h, spins_config))
print("--- my method %s---" % (time.time() - init))
print('\n\n')
init = time.time()
print('spins method:', spin_system.energy())
print("--- spins method %s---" % (time.time() - init))
print('\n\n')
init = time.time()
print('new github method:', 0.5 * np.dot(spins_config.transpose(), -J.dot(spins_config)) + np.dot(-h.transpose(), spins_config))
print("--- github method %s---" % (time.time() - init))

print(h)

# vector slicing

J = np.full(5-1, np.random.uniform(low= -2, high= 2, size=1))
print(J)
print(J[:-1])
print(np.diag(J, k=1))


# ISING MODEL 2D COUPLING

JJ = 1  # coupling
n = 2
m = 2
N = n*m  # number of spins
J = np.zeros((N,N))
spin_col_ind, spin_row_ind = np.meshgrid(range(n), range(m), indexing='ij')
print(spin_col_ind, '\n\n', spin_row_ind)

for i in range(n):
    for j in range(m):
        spin_ind = i*m + j
        J[spin_ind][i*m + (j+1)%m] += JJ if m>2 else JJ/2
        J[spin_ind][((i+1)%n)*m + j] += JJ if n>2 else JJ/2
        J[spin_ind][i*m + (j-1)%m] += JJ  if m>2 else JJ/2 # (0-h)%n == n-h
        J[spin_ind][((i-1)%n)*m + j] += JJ if n>2 else JJ/2

print(J)
print(np.transpose(np.nonzero(J)))

# test numpy

import numpy
anna = (1,2)

a, b = anna
print(1+ numpy.random.uniform(low= -2, high= 2, size=1))
print(type(numpy.random.uniform(low= -2, high= 2, size=1)))


# TEST class Ising Model()
from qmcmc_vqe_classes_V3_2 import *
n_spins=9
ising_model = IsingModel(n_spins=n_spins)
numpy.random.seed(4)

# print(ising_model.get_J(dimensionality='1D', random=True, nearest_neigh=False, plot=True, scale=2))
# plt.show()
# print('\n\n\n')
# print(ising_model.get_J(dimensionality='1D', random=False, nearest_neigh=False, plot=False, J_value=2))
# print('\n\n\n')
# print(ising_model.get_J(dimensionality='1D', random=False, nearest_neigh=False, plot=False))
# print('\n\n\n')
print(ising_model.get_J(dimensionality='2D', random=True, nearest_neigh=False, plot=True))
# plt.show()
print()
print(numpy.round(numpy.random.normal(loc=.0, scale=1, size=(n_spins, n_spins)), decimals=3))
# print('\n\n\n')
# print(ising_model.get_J(dimensionality='2D', random=False, nearest_neigh=False, plot=False, J_value=2))
# print('\n\n\n')
# print(ising_model.get_J(dimensionality='2D', random=False, nearest_neigh=False, plot=False))
# print('\n\n\n')
# print(ising_model.get_J(dimensionality='2D', random=False, nearest_neigh=True, plot=True, spin_grid=(3,3), J_value=3))
# plt.show()
# print('\n\n\n')
#print(ising_model.get_h(random=True, plot=True))


# test numpy

import numpy
anna = (1,2)

a, b = anna
print(1+ numpy.random.uniform(low= -2, high= 2, size=None))
print(type(numpy.random.uniform(low= -2, high= 2, size=None)))

# CHECK INDEXING 

for i in range(4):
            print('h:', i) # ham -= self.h[i] * pauli_list[i]
            for j in range(i):  # implementing new method here (taken from gitlab qmcmc)
                print('J:', i, j)  #ham -= self.J[i][j] * pauli_list[i] @ pauli_list[j]  

# HOW TO APPEND ARRAY TO PANDAS
import pandas as pd

df = pd.DataFrame()

for i in range(4):
    a = np.random.uniform(low=-2, high=2, size=5)
    df[f'a{i+1}'] = a.tolist()

print(df)
print(a)



# HOW TO APPEND ARRAY TO PANDAS
import pandas as pd

df = pd.DataFrame()

for i in range(4):
    a = np.random.uniform(low=-2, high=2, size=None)
    df = df.append({'a': a}, ignore_index=True)

print(df)

# CHECK spin system MAGNETISATION AND RANDOM STATE

from src.q_systems import SpinSystem
import numpy
n_spins=5
J = numpy.random.normal(loc=0.0, scale=1.0, size=(n_spins, n_spins))  # normal distributed as in the IBM paper
h = numpy.random.normal(loc=0.0, scale=1.0, size=n_spins)
spin_system = SpinSystem(n_spins, 10, J, h)

# print('spin config', spin_system.spins_configuration())
# print('spin config', spin_system.magnetization())
print(spin_system.decimal)
print(spin_system.statevector)
print()
print(spin_system.random_state())
print()
print(spin_system.decimal)
print(spin_system.statevector)



# CLASS INHERITANCE WITH __init__()

class num():
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def printa(self):
        print(self.a, self.b)

class num2(num):
    def __init__(self, a, b, c):
        super().__init__(a, b)
        self.c = c

obj = num2(1,2,3)

print(obj.b)


# CHECK IF INITIAL TRANSIENT CAN BE A PROBLEM
length = 5
a = np.zeros((length))
print(a[-length:], a)
print(a[-4:])


# ACCESS NUMPY ARRAY LENGTH
a = np.array([1,2,3,4,5])
print(type(a.size), '\n\n', a.size)
print(10e3


# tuples 
a = (1, 2)

b, c = a

print(c)

# arrays slicing

a = np.array([0,1,2,3,4,5,6,7,8,9])
print(a[0:9:2])


# SEABORN COLORMAPS

from qmcmc_vqe_classes_V4_2 import *
n_spins=9
ising_model = IsingModel(n_spins=n_spins)
J = ising_model.get_J(dimensionality='2D', random=True, nearest_neigh=False, plot=False)
# plotting colormap
plt.figure()
s = sns.heatmap(J, square=True, annot=False, cbar=True, cmap='viridis', xticklabels=[1,None, 2, None, 3, None, 4, None, 5], yticklabels=[1,None, 2, None, 3, None, 4, None, 5])
s.set_xlabel('$\gamma$', fontsize=35)
s.set_ylabel('$\\tau$', fontsize=35)
s.tick_params(labelsize=25)
# s.set(ylabel='$\\tau$', xlabel='$\gamma$')
# plt.yticks(rotation=0)
# plt.xticks(rotation=0)
plt.show()


# property decorator

# NON FUNZIONA COSI' , NON HAI UN SETTER
# class alpha():
#     def __init__(self, a):
#         self.a = a

#     @property
#     def a(self):
#         return self._a 

# cacao = alpha(3)

# print(cacao.a)
# cacao.a = 10

# print(cacao.a)

# COSI SI INVECE
class alpha():
    def __init__(self, a):
        self.a = a

    @property
    def get_a(self):
        return self.a 

cacao = alpha(3)

print(cacao.a)
cacao.a = 10

print(cacao.a)



# nome 
class alpha():
    def __init__(self, a, random=False):
        self.a = a
        self.name = '1D' + '_random_' + 'IsingModel'

    @property
    def get_a(self):
        return self.a 

cacao = alpha(1, True)
print(cacao.name)
cacca = alpha(2, False)
print(cacca.name)


# nome 
class alpha():
    def __init__(self, a):
        self.a = a
        self._name = self.build()

    @property
    def name(self):
        return self._name

    def build(self):
        return 'ccaaaca'

acca = alpha(1)

print(acca.name)

# 
class alpha():
    def __init__(self, a):
        self.a = a
        self.nome = 'aca'

class beta(alpha):
    def __init__(self, a, nom):
        super().__init__(a)
        self.nome = nom + self.nome

obj = beta(1, 'nnn')
print(obj.nome)

##
stringa = 'abcdefghilmnop'
print(stringa[3:])


# TEST class Ising Model 1D
from qmcmc_vqe_classes_V4_3 import *
n_spins=9
#numpy.random.seed(4)

# a = IsingModel_1D(n_spins, random=True, J_scale=2, h_scale=1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_1D(n_spins, random=False)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_1D(n_spins, random=False, J_value=1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_1D(n_spins, random=False, h_value=1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_1D(n_spins, random=True, J_scale=2, h_scale=1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

## 2D

# a = IsingModel_2D(n_spins, random=True)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_2D(n_spins, random=True, J_value=100, J_scale=3, J_loc=3, h_scale=1, h_loc=-1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_2D(n_spins, random=False, nearest_neigh=True, spins_grid=(3,3))
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_2D(n_spins, random=False, nearest_neigh=True, J_value=4, h_value=1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_2D(n_spins, random=False, nearest_neigh=False, J_value=22)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_2D(n_spins, random=False, h_value=1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

a = IsingModel_2D(n_spins, random=True)
print(a.J, '\n\n', a.h)
a.summary(plot=False)
print('\n\n\n')
print(a.name)


# converting arrays into strings
a = np.array([1, 2, 3])

stringa = f'alalalall{a}'

print(stringa)



# TEST class Ising Model NEW
from qmcmc_vqe_classes_V4_3 import *
n_spins=6
#numpy.random.seed(4)

# a = IsingModel_1D(n_spins, random=True, J_scale=2, h_scale=1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_1D(n_spins, random=False)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_1D(n_spins, random=False, J_value=1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_1D(n_spins, random=False, h_value=1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_1D(n_spins, random=True, J_scale=2, h_scale=1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

## 2D

# a = IsingModel_2D(n_spins, random=True)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_2D(n_spins, random=True, J_value=100, J_scale=3, J_loc=3, h_scale=1, h_loc=-1)
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

a = IsingModel_2D(n_spins, random=True, nearest_neigh=True, spins_grid=(3,2))
print(a.J, '\n\n', a.h)
a.summary(plot=True)
print('\n\n\n')

a = IsingModel_2D(n_spins, random=True, nearest_neigh=True, J_value=4, h_value=1, spins_grid=(3,2))
print(a.J, '\n\n', a.h)
a.summary(plot=True)
print('\n\n\n')

# a = IsingModel_2D(4, random=False, nearest_neigh=True, J_value=22, spins_grid=(2,2))
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

# a = IsingModel_2D(n_spins, random=False, nearest_neigh=True, J_value=1, spins_grid=(2,3))
# print(a.J, '\n\n', a.h)
# a.summary(plot=True)
# print('\n\n\n')

#
a = np.array([[1,2,3],
                [1,2,3],
                [1,2,3]])

print(np.tril(a, -1))



# # CHE TYPE SONO LE COLONNE DI DATAFRAME
# # Import pandas package
# import pandas as pd
  
# # Define a dictionary containing Students data
# data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
#         'Height': [5.1, 6.2, 5.1, 5.2],
#         'Qualification': ['Msc', 'MA', 'Msc', 'Msc']}
  
# # Convert the dictionary into DataFrame
# df = pd.DataFrame(data)
# df1 = pd.DataFrame()
  
# # Declare a list that is to be converted into a column
# address = ['Delhi', 'Bangalore', 'Chennai', 'Patna']
  
# # Using 'Address' as the column name
# # and equating it to the list
# df['Address'] = address
# df1['caso'] = df['Address']
# # df2 = df2.join(extracted_col)

# print(type(df1)

# dataframe
# sg_df = DataFrame()
# cf_df = DataFrame()  # index=range(5)

# print(cf_df)

# data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
#         'Height': [5.1, 6.2, 5.1, 5.2],
#         'Qualification': ['Msc', 'MA', 'Msc', 'Msc']}
  
# # Convert the dictionary into DataFrame
# df = DataFrame(data)
# # df_dict = df.to_dict()

# for i in range(4):
#     # cf_df = cf_df.join(df['Name'], rsuffix='_')
#     cf_df[f'Name_{i}'] = df['Name']

# print(cf_df)

# qualcosa coi dataframe
import pandas

sg_df = DataFrame(columns=['name', 'weight', 'height'])

data = {'name': 'babbabaj',
        'height': 4,
        'weight': 3}


for i in range(4):
    # cf_df = cf_df.join(df['Name'], rsuffix='_')
    sg_df = pandas.concat([sg_df, DataFrame([data])], axis=0, ignore_index=True)


print(sg_df)

# any()
a = np.zeros(4)
b = np.array([0,0,3,0])
print(a.any())
print(b.any())

'''

# copy arrays
a = np.array([0,0,3,0])
b = np.zeros(4)
print(a)
b = a
print(b)
b -= 1
print(a)