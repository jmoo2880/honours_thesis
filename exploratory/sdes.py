
from numpy import random 
import numpy as np

# standard wiener process for Discretised Brownian Motion
class Wiener():
    def simulate(T, N, seed=0):
        random.seed(seed)
        dt = T/N
        t = np.linspace(0, T, N)
        x = np.zeros(t.shape[0])
        x[0] = 0
        for i in range(1,len(t)):
            x[i] = x[i-1] + np.sqrt(dt) * np.random.normal()
        return t, x

