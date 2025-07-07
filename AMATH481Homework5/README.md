In this project, we examine how to solve Vorticity-Streamfunction equations numerically in Python. The time-evolution of the vorticity $\omega(x,y,t)$ and streamfunction $\psi(x,y,t)$ satisfy

(1) $\omega_t = -[\psi, \omega] + \nu \nabla^2 \omega$

(2) $\nabla^2 \psi = \omega$,

where $[\psi, \omega] = \psi_x \omega_y - \psi_y \omega_x, \nu \in \mathbb{R}$. 

We assume a Gaussian initial vorticity and attempt to solve the system of differential equations over the spatial domain $[-10,10] \times [-10, 10] \in \mathbb{R}^2$ from $t = 0$ to $t = 4$. We split the spatial domain into a $64 \times 64$ meshgrid over which we solve the differential equations numerically using time steps of $\Delta t = 0.5$. To do this, we construct matrices $A = \partial_x^2 + \partial_y^2$, $B = \partial_x$, and $C = \partial_y$ to represent the necessary derivative operators.

We then define a function \textbf{vorticity_rhs_{method}} that determines the right-hand side of (1) each time step:

1. Taking the current voriticty $\omega$ as an input, we solve for the current streamfunction $\psi$ from equation (2) using a predetermined method for that trial, which will be explained later.
2. With the current vorticity and resulting streamfunction, we use the derivative operators to calculate the necessary derivatives in (1), which can be used to determine $\omega_t$ at that time step.

With the function detailed above, we can now frame the problem in terms of a system of ordinary differential equations, which can be solved using \textbf{solve_ivp} from \textbf{scipy.integrate}.

The purpose of this exercise is to demonstrate the power of using FFT's in this context. Specifically, FFT's allow us to solve the streamfunction differential equation in Step 1. of \textbf{vorticity_rhs_fft with extreme efficiency. We examine the execution time when using FFT's in this context, compared to more crude and computationally expensive methods like \texttt{solve} and LU-decompositions. Even leveraging sparse matrices, more computationally efficient tools for matrix representation, resulted in much longer execution times.

We then plot the vorticity over time as well as generate an animation for a more satisfying view: [insert link].

Next, we experiment with different initial vorticies with our FFT solver on the streamfunction equation (2). We look at the bahvior of

1. Two oppositely "charged" Gaussian vorticies next to each other, i.e. with different signed amplitudes: [insert link]
2. Two same "charged" Gaussian vortices next to each other: [insert_link]
3. Two oppositely "charged" Gaussian vortices that collide: [insert link]
4. A random assortment of 10 vorticies with various different starting positions, amplitudes, strengths, charges, ellipticity: [insert link]
