import numpy as np
import matplotlib.pyplot as plt


# Defining the methods to solve the Dahlquist test ODE.

def Explicit_Euler_method(f, u_n, t_n, dt):
    u_new = u_n + dt * (f(t_n, u_n))
    return u_new

def Modified_Euler_method(f, u_n, t_n, dt):
    u_mid = u_n + (dt/2.0)*f(t_n, u_n)
    u_new = u_n + dt*f(t_n + dt/2.0, u_mid)
    return u_new

def Heun_method(f, u_n, t_n, dt):
    u_end = u_n + dt*f(t_n, u_n)
    u_new = u_n + (dt/2.0)*(f(t_n, u_n) + f(t_n + dt, u_end))
    return u_new


# defining the differential equations.

def main ():
    # setting up the constant values.
    lambda_1 = -10
    lambda_2 = -30
    h_1 = 0.13
    h_2 = 0.09
    h_3 = 0.029
    T = 20 
    N = 1000
    dt = T/N
    t = np.linspace(0, T, N+1)

    # Initializing the solutions

    u_Euler = np.zeros(N+1)
    u_Modified_Euler = np.zeros(N+1)
    u_Heun = np.zeros(N+1)

    # setting the initial conditions.
    u0 = 1
    u_Euler[0] = u0
    u_Modified_Euler[0] = u0
    u_Heun[0] = u0
    
    # defining the function for the differential equation.

    def f(t, u):
        f = lambda_1 * u
        return f
    

    for i in range(N):
        u_Euler[i+1] = Explicit_Euler_method(f,u_Euler[i], t[i], dt)
        u_Modified_Euler[i+1] = Modified_Euler_method(f, u_Modified_Euler[i], t[i], dt)
        u_Heun[i+1] = Heun_method(f, u_Heun[i], t[i], dt)

    plt.plot(t, u_Euler,color ='#444444' ,label = "Euler", marker = '.', linestyle ='--') # as per the format we have defined  color = k, line = --, in any order
    
    plt.plot(t, u_Modified_Euler,color = '#5a7d9a',label = "Modified_Euler", marker = 'o')
    plt.plot(t, u_Heun, color ='#adad3b', label = 'Heun', marker = '+')
    
    plt.title("Solutions")
    plt.xlabel("time steps")
    plt.ylabel("Solution")
    # plt.legend(['All dev', "python dev"]) # we need to pass a list of legends as per the order.
    plt.legend()
    plt.show()
    

if __name__ == '__main__':
    main()