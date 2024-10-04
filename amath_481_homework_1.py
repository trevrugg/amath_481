import numpy as np

# Problem 1.

## Newton-Raphson scheme

x_newt = np.array([-1.6])  ## initial guess
for i in range(1000):  ## iteration scheme, with maximum of 1000 iterations

    ## iteration formula, using f'(x_newt) = sin(3x) + 3xcos(3x) - exp(x_newt)
    x_newt = np.append(x_newt, x_newt[i] - (x_newt[i] * np.sin(3 * x_newt[i]) - np.exp(x_newt[i])) 
                  / (np.sin(3 * x_newt[i]) + 3 * x_newt[i] * np.cos(3 * x_newt[i]) - np.exp(x_newt[i])))

    fc = x_newt[i] * np.sin(3 * x_newt[i]) - np.exp(x_newt[i])   ## value of f at current x_newt

    if abs(fc) < 1e-6:   ## checking if we are within 1e-6 of 0
        break

## Bisection scheme 
x_mid = np.array([])  #initializing empty array of midpoint values

xl = -0.7; xr = -0.4  ## initial left and right endpoints
for i in range(1000):  ## iteration scheme, with maximum 1000 iterations
    
    xc = (xr + xl)/2  ## generating midpoint
    x_mid = np.append(x_mid, xc)  ## appending midpoint to array

    fc = xc * np.sin(3 * xc) - np.exp(xc)  ## calculate f at the current midpoint

    ## check if tolerance has been achieved
    if abs(fc) < 1e-6:
        break

    ## reassign left or right endpoint
    if(fc > 0):  
        xl = xc
    
    else:
        xr = xc

A1 = x_newt
A2 = x_mid
A3 = np.array([x_newt.size - 1, x_mid.size])

print(A1)
print(A1.size)

# Problem 2

## Initializing vectors and matrices

A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([1,0])
y = np.array([0,1])
z = np.array([1,2,-1])

## (a)
A4 = A + B

## (b)
A5 = 3 * x - 4 * y

## (c)
A6 = np.dot(A, x)

## (d)
A7 = np.dot(B, (x-y))

## (e)
A8 = np.dot(D, x)

## (f)
A9 = np.dot(D, y) + z

## (g)
A10 = np.dot(A, B)

## (h)
A11 = np.dot(B, C)

## (i)
A12 = np.dot(C, D)