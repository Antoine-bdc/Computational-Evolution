import main as m
import matplotlib.pyplot as plt
import time
import os
os.chdir("D:\\Documents\\Etudes\\Spore\\reproduction\\data ")
dir=os.getcwd()

## Parameters

grid_size = 50
duration = 10000
display = -1 # -1 = no display ; 0 = informations in the shell ; 2 = grid display
nb_agents = int((grid_size**2)/10)
nb_predators = int(nb_agents)
initial_probability=0
max_food = 100
food_regeneration=10 #In percentage of grid[i][j][2]
alpha = 0 #10
beta = 0 #1.4
growth="seasons"
period=1000
step=10


## Total food vs Nb agents correlation

smin,smax=2000,2020
daylist=[]
agent_list=[]
negative_sim=[]
negative_food_list=[]

# m.run_evolution(grid_size,duration,nb_agents,nb_predators,initial_probability,max_food,food_regeneration,alpha,beta,display,step,10,growth,period)
tot=0
for i in range (smin,smax):
    nombre_agents,day,food,nombre_predateurs,agent_list,predator_list,negativefood=m.run_evolution(grid_size,duration,nb_agents,nb_predators,initial_probability,max_food,food_regeneration,alpha,beta,display,step,i,growth,period)
    if abs(negativefood)>1:
        tot+=1
        negative_sim.append(i)
        negative_food_list.append(negativefood)
    print(negative_sim,negative_food_list)
print(tot,"/",smax-smin)








