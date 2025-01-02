# Lecture "Numerics of Differential Equations [MA 3301]" (WS 2024/25)
# @ TU Munich, School of Computation, Information and Technology, Dept. Mathematics
# by: Prof. B. Wohlmuth, Markus Muhr, Jonas Beddrich
#
# Draft solution implementation of the predator-prey model introduced in the lecture
# by means of three classical explicit numerical schemes: Euler, modified Euler and Heun.
# For an explanation of different numerical outcomes and observations to be made, see
# the accompanying draft solution document.


# IMPORTING PYTHON LIBRARIES
import numpy as np
import matplotlib.pyplot as plt


######################################################################

# DIFFERENT NUMERICAL SCHEMES
# each function implements a single step of the respective scheme
# going from time t_n to time t_n+1, i.e. computing the solution
# u_(n+1) from u_n

# Defining the global variables. Making them as globle functions.

def explicit_Euler_step(u_n, t_n, dt, f):
    u_new   = u_n + dt*f(t_n, u_n)
    return u_new

def modified_Euler_step(u_n, t_n, dt, f):
    u_tilde = u_n + (dt/2.0)*f(t_n, u_n)
    u_new   = u_n + dt*(f(t_n+dt/2.0, u_tilde))
    return u_new   

def Heun_step(u_n, t_n, dt, f):
    u_tilde = u_n + dt*f(t_n, u_n)
    u_new   = u_n + (dt/2.0)*(f(t_n, u_n) + f(t_n + dt, u_tilde))
    return u_new

print(2+4)
######################################################################

# Here everything we have defined will work locally.

# MAIN FUNCTION
def main():
    # SET UP OF THE PREDATOR PREY MODEL  
    # Parameters
    lambda_b = 1.0   # Reproduction rate of prey
    lambda_r = 1.5   # Fighting rate of predator
    b_e      = 0.5   # Equilibrium prey population
    r_e      = 0.25  # Equilibrium predator population
    b0       = 0.75  # Initial prey population
    r0       = 0.33  # Initial predator population
    # Right hand side function
    def f(t, u):
        f1 = lambda_b*u[0]*(1-u[1]/r_e)
        f2 = lambda_r*u[1]*(u[0]/b_e-1)
        f  = np.array([f1, f2])
        return f

    # SET UP FOR NUMERICAL SOLUTION PROCEDURE
    # Parameters
    T        = 30.0  # Final simulation time
    N        = 1000  # Number of time steps
    dt       = T/N   # Time step size
    # Time grid and solution arrays
    t           = np.linspace(0, T, N+1)  # Time grid
    u_Euler     = np.zeros((2, N+1))      # Solution array for Euler
    u_mod_Euler = np.zeros((2, N+1))      # Solution array for modified Euler
    u_Heun      = np.zeros((2, N+1))      # Solution array for Heun

    # NUMERICAL SOLUTION
    # Set initial values
    u_Euler[:,0]     = np.array([b0, r0])  # Euler method
    u_mod_Euler[:,0] = np.array([b0, r0])  # Modified Euler method
    u_Heun[:,0]      = np.array([b0, r0])  # Heun method
    # Time stepping loop
    for t_ind in range(N):
        # Compute the solution at the next time step
        u_Euler[:,t_ind+1]     = explicit_Euler_step(u_Euler[:,t_ind], t[t_ind], dt, f)
        u_mod_Euler[:,t_ind+1] = modified_Euler_step(u_mod_Euler[:,t_ind], t[t_ind], dt, f)
        u_Heun[:,t_ind+1]      = Heun_step(u_Heun[:,t_ind], t[t_ind], dt, f)

    # PLOTTING
    # Plot the populations over time
    plt.figure()
    plt.plot(t, u_Euler[0,:], 'g', label='Euler Prey')
    plt.plot(t, u_mod_Euler[0,:], 'g--', linewidth=2, label='Modified Euler Prey')
    plt.plot(t, u_Heun[0,:], 'g:', linewidth=5, label='Heun Prey') 
    plt.plot(t, u_Euler[1,:], 'r', label='Euler Predator')
    plt.plot(t, u_mod_Euler[1,:], 'r--', label='Modified Euler Predator')
    plt.plot(t, u_Heun[1,:], 'r:', linewidth=5, label='Heun Predator')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('Predator and Prey Populations')
    plt.grid(True)
    # Plotting in phase space
    plt.figure()
    plt.plot(u_Euler[0,:], u_Euler[1,:], 'b', label='Euler')
    plt.plot(u_mod_Euler[0,:], u_mod_Euler[1,:], 'b--', linewidth=2, label='Modified Euler')
    plt.plot(u_Heun[0,:], u_Heun[1,:], 'b:', linewidth=5, label='Heun')
    plt.xlabel('Prey Population')
    plt.ylabel('Predator Population')
    plt.legend()
    plt.title('Phase Space')
    plt.grid(True)
    # Show the plots
    plt.show()


######################################################################

if __name__ == '__main__':
    main()


"""
Why we use main functiojn ? 
To make the variables as the local variables we use the main function. 


"""