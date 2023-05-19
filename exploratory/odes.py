"""

An ensmble of common ODEs taken from Sprott unless otherwise stated
- ADD FUNCTION FOR NOISE (varying noise levels, distributions)
- ADD fn which returns list of systems for iterating through
- ADD fn which returns equations for comparison with SINDy recovered form 
"""

import numpy as np
from scipy.integrate import solve_ivp


############ Stochastic Processes ##############

# Lag One ARMA - fix the autoregressive coefficient phi
class ARMA():
    def simulate(phi, N, seed=42):
        np.random.seed(seed)
        theta = np.random.randn()
        x = np.zeros((N,1))
        eps = np.random.normal(size=(N,1))
        x[0] = np.random.randn()
        for i in range(1, N):
            x[i] = eps[i] + phi * x[i-1] + theta * eps[i - 1]
        return x

# Lag One AR Model
class AR1():
    def simulate(phi, N, seed=42):
        np.random.seed(seed)
        x = np.zeros(N)
        x[0] = np.random.uniform()
        eps = np.random.normal(size=N)
        for i in range(1, N):
            x[i] = eps[i] + phi * x[i-1]
        return x

# Lag Three AR Model - remove transient of length eta
class AR3():
    def simulate(phis, N, eta=100, seed=42): 
        phi1, phi2, phi3 = phis[0], phis[1], phis[2]
        np.random.seed(seed)
        x = np.zeros((N+eta,1))
        x[:3] = np.random.uniform(size=(3, 1))
        eps = np.random.normal(size=(N+eta,1))
        for i in range(3, N+eta):
            x[i] =  eps[i] + phi1 * x[i-1] + phi2 * x[i-2] + phi3 * x[i-3]
        return x[eta:]

class AR2():
    def simulate(phis, N, eta=100, seed=42): 
        phi1, phi2 = phis[0], phis[1]
        np.random.seed(seed)
        x = np.zeros((N+eta,1))
        x[:2] = np.random.uniform(size=(2, 1))
        eps = np.random.normal(size=(N+eta,1))
        for i in range(2, N+eta):
            x[i] =  eps[i] + phi1 * x[i-1] + phi2 * x[i-2] 
        return x[eta:]


################## Pendulums #########################
# Taken from John Taylor's Classical Mechanics
# State variable [theta, theta_dot]
# Gamma = Dimensionless Driving Force Strength
# Beta = Damping Constant
# omega_0 = natural frequency of pendulum
# omega = Driving Frequency

class DampedDrivenPendulum():
    def rhs(t, state, gamma, omega_0, omega, beta):
        phi, phi_dot = state[0], state[1]
        dxdt = phi_dot
        dvdt = gamma * omega_0 ** 2 * np.cos(omega * t) - 2 * beta * phi_dot - omega_0 **2 * np.sin(phi)
        return [dxdt, dvdt]


# Taken from https://scienceworld.wolfram.com/physics/DoublePendulum.html 
class DoublePendulum():
    def rhs(t, state, l1, l2, m1, m2, g):
        the1, the2, p1, p2 = state[0], state[1], state[2], state[3]

        d_the1 = (l2*p1 - l1*p2*np.cos(the1 - the2))/(l1**2*l2*(m1 + m2*np.sin(the1 - the2)**2))
        d_the2 = (l1*(m1 + m2)*p2 - l2*m2*p1*np.cos(the1 - the2))/(l1*l2**2*m2*(m1 + m2*np.sin(the1 - the2)**2))
        
        C1 = (p1 * p2 * np.sin(the1 - the2)) / (l1 * l2 * (m1 + m2 * np.sin(the1 - the2) ** 2))
        C2 = ((l2 ** 2* m2*p1**2 + l1**2*(m1 + m2)*p2**2 - l1*l2*m2*p1*p2*np.cos(the1 - the2))/(2*l1**2*l2**2*
        (m1 + m2*np.sin(the1 - the2)**2)**2))*(np.sin(2*(the1 - the2)))

        d_p1 = -(m1 + m2)*g*l1*np.sin(the1) - C1 + C2
        d_p2 = - m2 * g * l2 * np.sin(the2) + C1 - C2

        return [d_the1, d_the2, d_p1, d_p2]

################ Normal Forms ##################

# Taken From Sprott/Strogatz
class HopfNormalForm():
    def rhs(t, state, mu=0): # bifurcation point at mu = 0
        x, y = state[0], state[1]
        dxdt = - y + (mu - x ** 2 - y ** 2) * x
        dydt = x + (mu - x ** 2 - y ** 2) * y
        return [dxdt, dydt]
    def ic():
        return [0.2, - 0.6]


################ Oscillators #######################

# Taken from original SINDy paper examples, Appendix 4.1 (Supporting Information)
class CubicDampedOscilator():
    def rhs(t, state):
        x, y = state[0], state[1]
        dxdt = - 0.1 * x ** 3 + 2 * y ** 3
        dydt = - 2 * x ** 3 - 0.1 * y ** 3
        return [dxdt, dydt]
    def ic():
        return [2., 0.]
    
class ShawVanDerPol():
    def rhs(t, state):
        x, y = state[0], state[1]
        b = 1
        A = 1
        omega = 2
        dxdt = y + A * np.sin(omega * t)
        dydt = - x + b * (1 - x ** 2) * y
        return [dxdt, dydt]
    def ic():
        return [1.3, 0]

class VanDerPol():
    def rhs(t, state, A):
        x, y = state[0], state[1]
        dxdt = y
        dydt = - A * (x ** 2 - 1) * y - x
        return [dxdt, dydt]

class VanDerPolDriven():
    # A is amplitude of driving, mu is nonl-linear damping term, omega is angular freq of driving
    def rhs(t, state, mu, A, omega):
        x, y = state[0], state[1]
        dxdt = y
        dydt = mu * (1 - x ** 2) * y - x + A * np.sin(omega * t)
        return [dxdt, dydt]

class Brusselator():
    def rhs(t, state):
        x, y = state[0], state[1]
        a = 0.4
        b = 1.2
        A = 0.05
        omega = 0.8
        dxdt = x ** 2 * y - (b+1) * x + a + A * np.sin(omega*t)
        dydt = - x ** 2 * y + b * x
        return [dxdt, dydt]
    def ic():
        return [0.3, 2]

class Ueda():
    def rhs(t, state):
        x, y = state[0], state[1]
        b = 0.05
        A = 7.5
        omega = 1
        dxdt = y
        dydt = - x ** 3 - b * y + A * np.sin(omega*t)
        return [dxdt, dydt]
    def ic():
        return [2.5, 0.]

class DuffingTwoWell():
    def rhs(t, state):
        x, y = state[0], state[1]
        b = 0.25
        A = 0.4
        omega = 1
        dxdt = y
        dydt = - x ** 3 + x - b * y + A * np.sin(omega*t)
        return [dxdt, dydt]
    def ic():
        return [0.2, 0.]
    

class DuffingVanDerPol():
    def rhs(t, state):
        x, y = state[0], state[1]
        mu = 0.2
        gamma = 8
        A = 0.35
        omega = 1.02
        dxdt = y
        dydt = mu * (1 - gamma * x ** 2) * y - x ** 3 + A * np.sin(omega * t)
        return [dxdt, dydt]
    def ic():
        return [0.2, - 0.2]

class RayleighDuffing():
    def rhs(t, state):
        x, y = state[0], state[1]
        mu = 0.2
        gamma = 4
        A = 0.3
        omega = 1.1
        dxdt = y
        dydt = mu * (1 - gamma * y ** 2) * y - x ** 3 + A * np.sin(omega*t)
        return [dxdt, dydt]
    def ic():
        return [0.3, 0.]

########### Autonomous Dissipative Flows ###############

class Lorenz():
    def rhs(t, state, sigma, r, b):
        x, y, z = state[0], state[1], state[2]
        #sigma = 10
        #r = 28
        #b = 8/3
        dxdt = sigma * (y - x)
        dydt = - x * z + r * x - y
        dzdt = x * y - b * z
        return [dxdt, dydt, dzdt]
    def ic():
        return [0., -0.01, 9]

class Rossler():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        a = 0.2
        b = 0.2
        c = 5.7
        dxdt = - y - z
        dydt = x + a * y
        dzdt = b + z * (x - c)
        return [dxdt, dydt, dzdt]
    def ic():
        return [-9., 0, 0]

class DiffusionlessLorenz():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        R = 1
        dxdt = - y - x
        dydt = - x * z
        dzdt = x * y + R
        return [dxdt, dydt, dzdt]
    def ic():
        return [1., -1., 0.01]

class ComplexButterfly():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        a = 0.55
        dxdt = a * (y - z)
        dydt = - z * np.sign(x)
        dzdt = np.abs(x) - 1
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.2, 0.0, 0.0]

class Chen():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        a = 35
        b = 3
        c = 28
        dxdt = a * (y - x)
        dydt = (c - a) * x - x * z + c * y
        dzdt = x * y - b * z
        return [dxdt, dydt, dzdt]
    def ic():
        return [-10., 0, 28]

class HadleyCirculation():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        a = 0.25
        b = 5
        F = 8
        G = 1
        dxdt = - y ** 2 - z ** 2 - a * x + a * F
        dydt = x * y - b * x * z - y + G
        dzdt = b * x * y + x * z - z
        return [dxdt, dydt, dzdt]
    def ic():
        return [0., 0., 1.3]

class ACT():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        alpha = 1.8
        beta = - 0.07
        delta = 1.5
        mu = 0.02
        dxdt = alpha * (x - y)
        dydt = - 4 * alpha * y + x * z + mu * x ** 3
        dzdt = - delta * alpha * z + x * y + beta * z ** 2
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0., 0.]

class RabinovichFabrikant():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        gamma = 0.87
        alpha = 1.1
        dxdt = y * (z - 1 + x ** 2) + gamma * x
        dydt = x * (3 * z + 1 - x ** 2) + gamma * y
        dzdt = - 2 * z * (alpha + x * y)
        return [dxdt, dydt, dzdt]
    def ic():
        return [-1., 0, 0.5]

class RigidBodyMotion():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        alpha = 0.175
        dxdt = - 0.4 * x + y + 10 * y * z
        dydt = - x - 0.4 * y + 5 * x * y
        dzdt = alpha * z - 5 * x * y
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.6, 0., 0.]

class Chua():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        alpha = 9
        beta = 100/7
        a = 8/7
        b = 5/7
        dxdt = alpha * (y - x + b * x + 0.5 * (a - b) * (np.abs(x + 1) - np.abs(x - 1)))
        dydt = x - y + z
        dzdt = - beta * y
        return [dxdt, dydt, dzdt]
    def ic():
        return [0., 0., 0.6]

class MooreSpiegel():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        T = 6
        R = 20
        dxdt = y
        dydt = z
        dzdt = - z - (T - R + R * x ** 2) * y - T * x
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.1, 0., 0.]

class Thomas():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        b = 0.18
        dxdt = - b * x + np.sin(y)
        dydt = - b * y + np.sin(z)
        dzdt = - b * z + np.sin(x)
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.1, 0., 0.]

class Halvorsen():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        a = 1.27
        dxdt = - a * z - 4 * y - 4 * z - y ** 2
        dydt = - a * y - 4 * z - 4 * x - z ** 2
        dzdt = - a * z - 4 * x - 4 * y - x ** 2
        return [dxdt, dydt, dzdt]
    def ic():
        return [-5., 0., 0.]

class BurkeShaw():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        U = 10
        V = 13
        dxdt = - U * x - U * y
        dydt = - U * x * z - y
        dzdt = - U * x * y + V
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.6, 0., 0.]

class Rucklidge():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        kappa = 2
        lam = 6.7
        dxdt = - kappa * x + lam * y - y * z
        dydt = x 
        dzdt = - z + y ** 2
        return [dxdt, dydt, dzdt]
    def ic():
        return [1., 0., 4.5]

class WINDMI():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        a = 0.7
        b = 2.5
        dxdt = y
        dydt = z
        dzdt = - a * z - y + b - np.exp(x)
        return [dxdt, dydt, dzdt]
    def ic():
        return [0., 0.8, 0.]

class Quadratic():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        a = 2.017
        dxdt = y
        dydt = z
        dzdt = - a * z + y ** 2 - x
        return [dxdt, dydt, dzdt]
    def ic():
        return [- 0.9, 0., 0.5]

class Cubic():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        a = 2.028
        dxdt = y
        dydt = z
        dzdt = - a * z + x * y ** 2 - x
    def ic():
        return [0., 0.96, 0.]

class PieceWise():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        a = 0.6
        dxdt = y
        dydt = z
        dzdt = - a * z - y + np.abs(x) - 1
        return [dxdt, dydt, dzdt]
    def ic():
        return [0., - 0.7, 0.]

class DoubleScroll():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        a = 0.8
        dxdt = y
        dydt = z
        dzdt = - a * (z + y + x - np.sign(x)) # sign function NOT sin
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.01, 0.01, 0.]


# Sprott's Simple 3D chaotic Flows
# Parameters are baked into differential equations
class SprottA():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = y
        dydt = - x + y * z
        dzdt = 1 - y ** 2
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottB():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = y * z
        dydt = x - y 
        dzdt = 1 - x * y
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottC():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = y * z
        dydt = x - y 
        dzdt = 1 - x ** 2
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottD():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = - y
        dydt = x + z 
        dzdt = x * z + 3 * y ** 2
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottE():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = y * z
        dydt = x ** 2 - y 
        dzdt = 1 - 4 * x
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottF():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = y + z 
        dydt = - x + 0.5 * y 
        dzdt = x ** 2 - z
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottG():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = 0.4 * x + z 
        dydt = x * z - y 
        dzdt = - x + y
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottH():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = - y + z ** 2
        dydt = x + 0.5 * y 
        dzdt = x - z
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottI():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = - 0.2 * y
        dydt = x + z
        dzdt = x + y ** 2 - z
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottJ():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = 2 * z
        dydt = - 2 * y + z
        dzdt = - x + y + y ** 2
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottK():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = x * y - z
        dydt = x - y
        dzdt = x + 0.3 * z
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottL():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = y + 3.9 * z
        dydt = 0.9 * x ** 2 - y 
        dzdt = 1 - x 
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottM():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = - z
        dydt = - x ** 2 - y
        dzdt = 1.7 + 1.7 * x + y 
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottN():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = - 2 * y
        dydt = x + z ** 2
        dzdt = 1 + y - 2 * z
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottO():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = y
        dydt = x - z
        dzdt = x + x * z + 2.7 * y 
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottP():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = 2.7 * y + z
        dydt = - x + y ** 2
        dzdt = x + y
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottQ():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = - z 
        dydt = x - y 
        dzdt = 3.1 * x + y ** 2 + 0.5 * z
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottR():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = 0.9 - y
        dydt = 0.4 + z
        dzdt = x * y - z
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions

class SprottS():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = - x - 4 * y
        dydt = x + z ** 2
        dzdt = 1 + x
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.5, 0.5, 0.5] # default initial conditions


######### Conservative Flows #############

class DrivenPendulum():
    def rhs(t, state):
        x, y = state[0], state[1]
        A = 1.0
        omega = 0.5
        dxdt = y
        dydt = - np.sin(x) + A * np.sin(omega * t)
        return [dxdt, dydt]
    def ic():
        return [0., 0.]

class DrivenChaotic():
    def rhs(t, state):
        x, y = state[0], state[1]
        omega = 1.88
        dxdt = y
        dydt = - x ** 3 + np.sin(omega * t)
        return [dxdt, dydt]
    def ic():
        return [0., 0.]

class NoseHoover():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        a = 1
        dxdt = y
        dydt = - x + y * z
        dzdt = a - y ** 2
        return [dxdt, dydt, dzdt]
    def ic():
        return [0., 5., 0.]

class Labyrinth():
    def rhs(t, state):
        x, y, z = state[0], state[1], state[2]
        dxdt = np.sin(y)
        dydt = np.sin(z)
        dzdt = np.sin(x)
        return [dxdt, dydt, dzdt]
    def ic():
        return [0.1, 0., 0.]

class HenonHeiles():
    def rhs(t, state):
        x, y, v, w = state[0], state[1], state[2], state[3]
        dxdt = v
        dydt = w
        dvdt = - x - 2 * x * y
        dwdt = - y - x ** 2 + y ** 2 
        return [dxdt, dydt, dvdt, dwdt]
    def ic():
        return [0.499, 0., 0., 0.03160676]


############# Neuro ##############

# Taken from Brain Dynamics Toolkit
class WilsonCowan():
    def rhs(t, state, 
    wee, wei, wie, wii, be, bi, 
    Je, Ji, taue, taui):
        E, I = state[0], state[1]
        F = lambda v: 1 / (1 + np.exp(-v))
        dEdt = 1/taue * (- E + F(wee * E - wei * I - be + Je))
        dIdt = 1/taui * (- I + F(wie * E - wii * I - bi + Ji))
        return [dEdt, dIdt]
    

