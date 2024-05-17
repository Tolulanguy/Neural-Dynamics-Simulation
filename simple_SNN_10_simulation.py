import numpy as np
import matplotlib.pyplot as plt
num_neurons = 10 #number of neurons
T = 10 #Duration of simulation
spike_rate = 0.1 #spike/firing rate in spikes per second for each neuron
num_steps = T * 1000 #converting duration to milliseconds (ms) as the chosen unit
spike_train_data = np.zeros((num_neurons, num_steps))

#initialize an array for storing the spike train data for each neuron
for neuron in range(num_neurons): #iterating for each neuron until completed
    for step in range(num_steps):
        #iterate for each time_steps to generate spikes within each neuron we are iterating.
        if np.random.rand() < (spike_rate/1000):
            #spike rate converted from spike/second to spike/millisecond
            spike_train_data[neuron, step] = 1
print("Spike train data for the first neuron: ")
print(spike_train_data)
membrane_potential = np.zeros((num_neurons)) #initialize membrane potential of neurons
threshold = 1.0 #define membrane potential threshold for spike generation by neurons
for step in range(num_steps): #simulate dynamics over time
    for neuron in range(num_neurons): #iterate over each neuron
        membrane_potential[neuron] += spike_train_data[neuron, step]
        #update membrane potential based on the input spikes
        if (membrane_potential[neuron] >= threshold): #checking if membrane potential hits threshold so that output spike can be generated
            print(f"Neuron {neuron} spiked at time {step} ms") #generating output spike
            membrane_potential[neuron] = 0.0

#Plotting a graph to show analysis of spike rate of each neuron
spike_rates = np.sum(spike_train_data, axis=1)/T #calculating spike rate for each neuron
#histogram graph
plt.figure(figsize=(8, 6))
plt.hist(spike_rates, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Distribution of Spike Rates')
plt.xlabel('Spike Rate - spikes per second (Hz)')
plt.ylabel('Number of Neurons')
plt.grid(True)
plt.show()

#spike rate analysis with line graph
plt.figure(figsize=(8, 6))
for neuron_idx in range(num_neurons):
    plt.plot(num_steps, spike_rates[neuron_idx], label=f'Neuron{neuron_idx + 1}')
plt.title('Spike Rates of Individual Neurons Over Time')
plt.xlabel('TIme Stamps')
plt.ylabel('Spike Rates - spikes per second (Hz)')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#Correlation analysis to assess potential functional relationships between neurons based on correlation between their spike rates.
import pandas as pd
import seaborn as sns
spike_rates_df = pd.DataFrame(spike_rates) #converts numpy array to Panda DataFrame because corr matrix is not compatible with numpy array except panda dataframe
correlation_matrix = spike_rates_df.corr() #compute correlation matrix. adding _df converts the numpy array to pandas dataframe
#plotting heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Spike Rates')
plt.xlabel('Neurons')
plt.ylabel('Neurons')
plt.show()
