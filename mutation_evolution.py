import numpy as np
import random
import matplotlib.pyplot as plt

def mutate(value,probability):   # probability is between [0,1]
    if random.random()<probability:
        return(value+(random.random()-0.5)*value/2) # Uniform random mutation between 75%-125% of initial value
    else :return(value)

probability=0.50
duration=10000
nb=100000
 
tab=[0]*duration
for test in range(nb):
    run=[probability]
    for i in range(duration-1):
        run.append(mutate(run[i],run[i]))
    for i in range(duration): #averaging all the simulations
        tab[i]+=run[i]

for i in range(duration):
    tab[i]/=nb 
    # tab[i]=np.log(tab[i])


plt.plot(range(duration),tab,label=str(nb)+" averaged simulations")
plt.xscale('log')
plt.legend()
plt.show()
plt.plot(range(duration),run,label="single simulation")
plt.legend()
plt.xscale('log')
plt.show()