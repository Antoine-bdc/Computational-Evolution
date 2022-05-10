import matplotlib.pyplot as plt
import numpy as np
import pickle 


def average(smin,smax,daymin):
    T,avg_grid_food,a_avg_nb_agents,a_avg_food,a_avg_min_food,a_avg_nb_offspring,a_avg_food_trans,a_avg_mut_prob,a_avg_generation,p_avg_nb_agents,p_avg_food,p_avg_min_food,p_avg_nb_offspring,p_avg_food_trans,p_avg_mut_prob,p_avg_generation,std_grid_food,a_std_nb_agents,a_std_food,a_std_min_food,a_std_nb_offspring,a_std_food_trans,a_std_mut_prob,a_std_generation,p_std_nb_agents,p_std_food,p_std_min_food,p_std_nb_offspring,p_std_food_trans,p_std_mut_prob,p_std_generation = pickle.load(open('simulations_'+str(smin)+"_"+str(smax)+'.pickle', "rb" ))
    T,nb_sim,grid_food,a_nb_agents,a_food,a_min_food,a_nb_offspring,a_food_trans,a_mut_prob,a_generation,p_nb_agents,p_food,p_min_food,p_nb_offspring,p_food_trans,p_mut_prob,p_generation = pickle.load(open('raw_simulations_'+str(smin)+"_"+str(smax)+'.pickle', "rb" ))
    
    file = open("simulation_"+str(smin)+"/datafile.txt","r")
    for line in file:
        dat = line.split(" ")
        if dat[0]=="step":step=int(dat[1])
    file.close()
    
    daymin=daymin//step
    
    print("simulations ",smin,"-",smax)
    print("avg food",np.mean(avg_grid_food[daymin:]),"pm",np.std(avg_grid_food[daymin:]))
    print("avg pop",np.mean(a_avg_nb_agents[daymin:]),"pm",np.std(a_avg_nb_agents[daymin:]))
    print("avg agent food",np.mean(a_avg_food[daymin:]),"pm",np.std(a_avg_food[daymin:]))
    
    print("")
    
    all_grid_food=[]
    all_a_nb_agents=[]
    all_a_food=[]
    for i in range(len(grid_food[daymin:])):
        for j in range(smax-smin):
            all_grid_food.append(grid_food[i][j])
            all_a_nb_agents.append(a_nb_agents[i][j])
    for i in range(len(grid_food[daymin:])):   
        for j in range(len(a_food[i])):
            all_a_food.append(a_food[i][j])
    print("advanced average")
    print("avg food",np.mean(all_grid_food),"pm",np.std(all_grid_food))
    print("avg pop",np.mean(all_a_nb_agents),"pm",np.std(all_a_nb_agents))
    print("avg agent food",np.mean(all_a_food),"pm",np.std(all_a_food))


smin,smax=60,80
daymin=100

average(smin,smax,daymin)


