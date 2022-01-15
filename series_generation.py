import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
rng = default_rng(2022)

N_values = 10**5
indices = np.arange(N_values)

opt = input("Series to generate (O-U/LOG/UNI): ")

# Uniform distribution:

if opt == 'UNI':
    uniform = rng.random([N_values])

    np.savetxt('uniform_ts.dat', np.column_stack((indices, uniform)))

    print('Uniform distribution time series succesfully generated.')

# Ornstein - Uhlenbeck process:

if opt == 'O-U':
    theta = 1
    sigma = 1
    x0uhl = 1
    or_uhl = np.zeros(N_values)
    or_uhl[0] = x0uhl

    for i in range(N_values-1):
        or_uhl[i+1] = or_uhl[i] - theta*or_uhl[i] + sigma * rng.standard_normal()

    np.savetxt('or_uhl_ts.dat', np.column_stack((indices, or_uhl)))

    print('Ornstein - Uhlenbeck process time series succesfully generated.')

# Logistic map:

if opt == 'LOG':
    r = 4
    x0log = .4
    logmap = np.zeros(N_values)
    n = 0

    while n < 10000:
        x0log = r*x0log*(1-x0log)
        n += 1

    logmap[0] = x0log

    for i in range(N_values-1):
        logmap[i+1] = r*logmap[i]*(1-logmap[i])

    np.savetxt('logmap_ts.dat', np.column_stack((indices, logmap)))

    print('Logistic map time series succesfully generated.')