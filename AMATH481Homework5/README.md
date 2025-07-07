In this project, we examine how to solve Vorticity-Streamfunction equations numerically in Python. The time-evolution of the vorticity $\omega(x,y,t)$ and streamfunction $\psi(x,y,t)$ satisfy

(1) \omega_t + [\psi, \omega] = \nu \nabla^2 \omega

(2) \nabla^2 \psi = \omega,

where $[\psi, \omega] = \psi_x \omega_y - \psi_y \omega_x.
