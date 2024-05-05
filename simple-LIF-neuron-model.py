#simulating the dynamics of a simple Leaky Integrate-and-Fire Neuron model

import numpy as np
import matplotlib.pyplot as plt
#R is Membrane resistance and C is Membrane Capacitance initial values
R = 1
C = 1
#I_in reps input current
I_in = 1
#dt is time step in seconds; reps duration of each time increment in simulation
#T represents total simulation time. dt and T determines accuracy
dt = 0.01 #smaller dt = more accuracy
T = 10
num_steps = int(T/dt) #number of time steps
V = 0 #initial membrane potential at the start of simulation
membrane_potential = [] #a default list to store the values at each simulation
time_values = []
for step in range(num_steps):
    dVdt = (1/C)*(-V/R + I_in)
    V += dVdt * dt
    #For loop to iterate for each simulation until completed. For each simulation, differential equation for LIF neuron model is used to calculate rate of change of V. Eular's model is then used to updaqte V
    membrane_potential.append(V)
    time_values.append(step * dt)
    #V and time values are stored for each iteration.
plt.plot(time_values, membrane_potential)
plt.xlabel('Time (s)')
plt.ylabel('Membrane Potential (V)')
plt.title('Membrane Potential of LIF Neuron')
plt.grid(True)
plt.show()
