import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
from scipy.integrate import odeint
from scipy.integrate import RK45


# Part B

N = 81
dx = 0.1
L= 4

x_span = [-L, L]
x_partition = np.linspace(x_span[0], x_span[1], (x_span[1] - x_span[0]) * 10 + 1)

# Creating matrix A
A = np.zeros((N-2, N-2))
for j in range(0, N - 2):
    A[j,j] = -2 - ((x_partition[j + 1]) ** 2 * 0.1**2)

for j in range(0, N-3):
    A[j+1, j] = 1
    A[j, j + 1] = 1

A[0,0] = -(2/3) - (x_partition[1]**2 * 0.1**2)
A[N-3,N-3] = -(2/3) - (x_partition[N-2]**2 * 0.1**2)
A[0,1] = 2/3
A[N-3, N - 4] = 2/3

# Dividing by -dx^2
A = A / (-(0.1)**2)

# Finding first five eigenvalues and eigenvectors
A4, A3 = eigs(A, k =5, which = "SM")
A4 = A4.real
A3 = A3.real
A3 = np.append(A3, [[(4/3) * A3[N-3][0] - (1/3) * A3[N-4][0],
                     (4/3) * A3[N-3][1] - (1/3) * A3[N-4][1],
                     (4/3) * A3[N-3][2] - (1/3) * A3[N-4][2],
                     (4/3) * A3[N-3][3] - (1/3) * A3[N-4][3],
                     (4/3) * A3[N-3][4] - (1/3) * A3[N-4][4]]], axis = 0)
A3 = np.append([[(4/3) * A3[0][0] - (1/3) * A3[1][0],
                     (4/3) * A3[0][1] - (1/3) * A3[1][1],
                     (4/3) * A3[0][2] - (1/3) * A3[1][2],
                     (4/3) * A3[0][3] - (1/3) * A3[1][3],
                     (4/3) * A3[0][4] - (1/3) * A3[1][4]]], A3, axis = 0)

col = ['r', 'b', 'g', 'c', 'm']

# Normalization of eigenfunctions

#plt.figure()
for j in range(5):
    norm = np.trapz(A3[:, j] * A3[:, j], x_partition)
    A3[:, j] = abs(A3[:, j] / np.sqrt(norm))
    #plt.plot(x_partition, A3[:, j], col[j])

def shoot_algorithm(y, x, epsilon, gamma):
    return [y[1], gamma * abs(y[0])**2 * y[0] + (x**2 - epsilon)*y[0]]

A6 = np.zeros(2)
A5 = np.zeros((81,2))
A8 = np.zeros(2)
A7 = np.zeros((81,2))
tol = 1e-4

# Part C

for gamma in [-0.05, 0.05]:
    epsilon_start = 0.1
    #plt.figure()
    # Inner for-loop finding each of the 5 modes
    for modes in range(1,3):
        A = 0.001   # Reset A each time
        dA = 0.001
        epsilon = epsilon_start
        d_epsilon = 0.01
        for a in range(1000):
            # Inner-for loop finding epsilon for each mode
            for b in range(1000):
                phi0 = [A, A * np.sqrt(L**2 - epsilon)]
                phi = odeint(shoot_algorithm, phi0, x_partition, args = (epsilon, gamma))
                if abs(phi[-1, 1] + np.sqrt(L**2 - epsilon) * phi[-1,0]) < tol:
                    break
                if (-1)**(modes + 1) * (phi[-1,1] + np.sqrt(L**2 - epsilon) * phi[-1,0]) > 0:
                    epsilon += d_epsilon
                else:
                    epsilon -= d_epsilon / 2
                    d_epsilon /= 2
            norm = np.abs(np.trapz(phi[:, 0] * phi[:, 0], x_partition))
            epsilon_start = epsilon + 0.1
            if abs(norm - 1) < tol:
                if gamma == -0.05:
                    A7[:, modes - 1] = np.abs(phi[:, 0]) / np.sqrt(norm)
                    A8[modes - 1] = epsilon
                else:
                    A5[:, modes - 1] = np.abs(phi[:,0]) / np.sqrt(norm)
                    A6[modes - 1] = epsilon
                break
            if norm > 1:
                A -= dA
                dA /= 2
            else:
                A += dA

        

plt.plot(x_partition, A5[:, 0], col[0])
plt.plot(x_partition, A5[:, 1], col[1])
plt.plot(x_partition, A7[:, 0], col[2])
plt.plot(x_partition, A7[:, 1], col[3])
plt.show()










    

