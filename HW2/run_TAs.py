from re import T
import numpy as np
from maxwell1d import maxwell1d
import matplotlib.pyplot as plt

###############################################################################
# MECA2300 - UCLouvain
# HW2 : Maxwell's equation in 1D
# This is a basic function to test your implementation of HW2. It should return 
# a plot showing the solution at different time-steps.
# Authors : Pierre Jacques, Matteo Couplet, Thomas Leyssens
###############################################################################

# Inputs:
L = 384400e3                            # distance between Earth and Moon [m]
E0 = lambda x,L : 0.                    # initial condition of q.2
H0 = lambda x,L : np.exp(-(10*x/L)**2)  # initial condition of q.2
n = 25                                  # space discretization
eps = np.ones(2*n)*8.854e-12            # vacuum permittivity [F/m]
mu = np.ones(2*n)*4*np.pi*1e-7          # vacuum permeability [H/m]
c = (eps[0]*mu[0])**(-1/2)              # speed of light [m/s]
Z0 = np.sqrt(mu[0] / eps[0])            # vacuum impedance [Ohm]
h = L/n                                 # space discretization
dt = 0.001                              # time step
CFL = c*dt/h                            # CFL number
Tfinal = 2*L/c                          # final time
m = int(Tfinal/dt)                      # number of time steps
p = 3                                   # order of Legendre polynomial
rktype = "RK44"                         # time integration type ("ForwardEuler", "RK22", "RK44")                
bctype = "reflective"                   # boundary condition type ("periodic", "reflective", "non-reflective")
print("CFL : ", CFL)

# Computation of the solution
sol = maxwell1d(L, E0, H0, n, eps, mu, dt, m, p, rktype, bctype)

# Computation of the legendre polynomials up to order p to evaluate the solution.
def leg(x):
    P = np.ones(p+1)
    P[0] = 1
    if p >= 1: P[1] = x
    for i in range(1, p):
        P[i+1] = ((2*i+1)*x*P[i] - i*P[i-1]) / (i+1)
    return P

# Sampling of solution: 
# returns the value of E or H at N points within each element
def sample_solution(uhat, N):   
    sol = np.zeros((2*n, N+1))
    r = np.linspace(-1, 1, N+1)
    x = np.zeros((2*n, N+1))
    for e in range(2*n):
        x[e,:] = -L + h*e + (1+r)/2 * h # affine map
        for j in range(N+1):
            Pr = leg(r[j])
            for i in range(p+1):
                sol[e,j] += Pr[i] * uhat[i,e]
    return x,sol

# Plots:
N = 2*p
t_samples = [0, int(m/4), int(m/2), int(3*m/4)]
x = np.zeros((len(t_samples), 2*n, N+1))
solE = np.zeros((len(t_samples), 2*n, N+1))
solH = np.zeros((len(t_samples), 2*n, N+1))
for i,t in enumerate(t_samples) :
    x[i,:,:], solE[i,:,:] = sample_solution(sol[0,:,:,t], N)
    x[i,:,:], solH[i,:,:] = sample_solution(sol[1,:,:,t], N)

# E
fig1, axs1 = plt.subplots(4)
fig1.suptitle("1D Maxwell's equations : E")
for i,t in enumerate(t_samples):
    for e in range(2*n):
        axs1[i].plot(x[i,e,:], solE[i,e,:], color='b')
        axs1[i].set(ylabel="$E(t=%.3f)$"%(t*dt))
axs1[-1].set(xlabel="x")      

# H
fig2, axs2 = plt.subplots(4)
fig2.suptitle("1D Maxwell's equations : H")
for i,t in enumerate(t_samples):
    for e in range(2*n):
        axs2[i].plot(x[i,e,:], solH[i,e,:]*Z0, color='r')
        axs2[i].set(ylabel="$Z_0*H(t=%.3f)$"%(t*dt))
axs2[-1].set(xlabel="x")      

plt.show()

