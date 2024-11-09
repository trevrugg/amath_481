import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
import matplotlib.pyplot as plt

# Want to solve differential equation of the form: y'' = (x^2 - epsilon)y
# We can set this up as system of differential equations, where
# y_1 = y'
# y_2 = y''

# Then, our differential equation becomes the system
# y_1' = y_2
# y_2' = (x^2 - epsilon)y_1

# Defining the shooting algorithm using the system of differential
# equations defined above.

def shoot_algorithm(y, x, epsilon):
    return [y[1], ((x**2) - epsilon)*y[0]]

tol = 1e-4
col = ['r', 'b', 'g', 'c', 'm']
L = 4 

# Defining our interval of x-values
xspan = [-L, L]

# Defining our partition of our interval [-L, L]
x_partition = np.linspace(xspan[0], xspan[1], (xspan[1] - xspan[0]) * 10 + 1)

# Initializing our vector of eigenvalues and matrix of eigenvectors
epsilon_vector = np.zeros(5)
solution_vector = np.zeros((81, 5))

epsilon_start = 0.1

# Inner for-loop finding each of the 5 modes
for modes in range(1,6):
    epsilon = epsilon_start
    d_epsilon = 0.01
    # Inner-for loop finding epsilon for each mode
    for i in range(1000):
        phi0 = [1, np.sqrt(L**2 - epsilon)]
        phi = odeint(shoot_algorithm, phi0, x_partition, args = (epsilon,))
        if abs(phi[-1, 1] + np.sqrt(L**2 - epsilon) * phi[-1,0]) < tol:
            epsilon_vector[modes - 1] = epsilon
            break
        if (-1)**(modes + 1) * (phi[-1,1] + np.sqrt(L**2 - epsilon) * phi[-1,0]) > 0:
            epsilon += d_epsilon
        else:
            epsilon -= d_epsilon / 2
            d_epsilon /= 2
    epsilon_start = epsilon + 0.1
    norm = np.trapz(phi[:, 0] * phi[:, 0], x_partition)
    solution_vector[:, (modes - 1)] = abs(phi[:, 0] / np.sqrt(norm))
    #plt.plot(x_partition, abs(phi[:, 0] / np.sqrt(norm)), col[modes - 1])
#plt.show()

A1 = solution_vector
A2 = epsilon_vector

