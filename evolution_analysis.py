import matplotlib.pyplot as plt
import numpy as np
import pickle 
import numpy.fft as fft
from matplotlib.font_manager import FontProperties
from matplotlib import colors
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
import time as t


mult=1
data=0

def weighted_average(smin,smax,parameters=["plot"],variable="a_food"):
    """outputs plots and/or histograms of different genes of the simulation over time"""
    
    plot,hist,override,animate,hist2d=0,0,0,0,0
    if "plot" in parameters:plot=1
    if "override" in parameters:override=1
    if "hist" in parameters: hist=1
    if "animate" in parameters: animate=1
    if "2dhist" in parameters: hist2d=1
    global mult
    
    ## Loading data
    try:
        if override==1:
            print("Overriding data")
            A=[]
            A[0]+=1
        T,avg_grid_food,a_avg_nb_agents,a_avg_food,a_avg_min_food,a_avg_nb_offspring,a_avg_food_trans,a_avg_mut_prob,a_avg_generation,p_avg_nb_agents,p_avg_food,p_avg_min_food,p_avg_nb_offspring,p_avg_food_trans,p_avg_mut_prob,p_avg_generation,std_grid_food,a_std_nb_agents,a_std_food,a_std_min_food,a_std_nb_offspring,a_std_food_trans,a_std_mut_prob,a_std_generation,p_std_nb_agents,p_std_food,p_std_min_food,p_std_nb_offspring,p_std_food_trans,p_std_mut_prob,p_std_generation = pickle.load(open('simulations_'+str(smin)+"_"+str(smax)+'.pickle', "rb" ))
        T,nb_sim,grid_food,a_nb_agents,a_food,a_min_food,a_nb_offspring,a_food_trans,a_mut_prob,a_generation,p_nb_agents,p_food,p_min_food,p_nb_offspring,p_food_trans,p_mut_prob,p_generation = pickle.load(open('raw_simulations_'+str(smin)+"_"+str(smax)+'.pickle', "rb" ))
        print("Loading data")
    
    ## Extracting data 
    except:
        if override!=1:print("Extracting data")
        durationmax=0
        for sim in range(smin,smax):
            print("data/simulation_"+str(sim)+"/datafile.txt")
            file = open("data/simulation_"+str(sim)+"/datafile.txt","r")
            for line in file:
                dat = line.split(" ")
                if dat[0]=="duration":duration=int(dat[1])
                if dat[0]=="step":step=int(dat[1])
                if dat[0]=="max_food":max_food=float(dat[1])
                if dat[0]=="n":n=int(dat[1])
                if dat[0]=="initial_probability":initial_probability=float(dat[1])
            print(step)
            file.close()    
            if duration>durationmax:simax=sim
            durationmax=max(durationmax,duration)
        T=[]
        nb_sim=[]
        grid_food=[]
        #agents:
        a_nb_agents=[]
        a_food  =[]
        a_min_food =[]
        a_nb_offspring =[]
        a_food_trans =[]
        a_mut_prob =[]
        a_generation = []
        #predators:
        p_nb_agents=[]
        p_food  =[]
        p_min_food =[]
        p_nb_offspring =[]
        p_food_trans =[]
        p_mut_prob =[]
        p_generation= []
        for i in range(0,durationmax//step):
            T+=[i*step]
            nb_sim+=[0]
            grid_food+=[[]]
            #agents:
            a_nb_agents+=[[]]
            a_food  +=[[]]
            a_min_food +=[[]]
            a_nb_offspring +=[[]]
            a_food_trans +=[[]]
            a_mut_prob +=[[]]
            a_generation +=[[]]
            #predators:
            p_nb_agents+=[[]]
            p_food  +=[[]]
            p_min_food +=[[]]
            p_nb_offspring +=[[]]
            p_food_trans +=[[]]
            p_mut_prob +=[[]]
            p_generation +=[[]]
            
            
        for sim in range(smin,smax):
            file = open("data/simulation_"+str(sim)+"/datafile.txt","r")
            for line in file:
                dat = line.split(" ")
                if dat[0]=="duration":duration=int(dat[1])
            file.close()
            for i in range(0,duration//step):
                if (1*i%100)==0:print("sim",sim,":",round(step*100*i/duration,2))
                file = open("data/simulation_"+str(sim)+"/"+str(i*step)+".dat","r")
                a_nb=0
                p_nb=0
                grid_food_local=0
                for line in file:
                    dat = line.split(" ")
                    if dat[0]=="a":
                        a_nb+=1
                        a_food[i] += [float(dat[3])]
                        a_min_food[i] += [float(dat[5])]
                        a_nb_offspring[i] += [float(dat[6])]
                        a_food_trans[i] += [float(dat[7])]
                        a_mut_prob[i] += [float(dat[8])]
                        a_generation[i] += [float(dat[10])]
                    if dat[0]=="p":
                        p_nb+=1
                        p_food[i] += [float(dat[3])]
                        p_min_food[i] += [float(dat[5])]
                        p_nb_offspring[i] += [float(dat[6])]
                        p_food_trans[i] += [float(dat[7])]
                        p_mut_prob[i] += [float(dat[8])]
                        p_generation[i] += [float(dat[11])]
                    if dat[0]=="f":
                        grid_food_local+=float(dat[3])
                a_nb_agents[i]+=[a_nb]
                p_nb_agents[i]+=[p_nb]
                grid_food[i]+=[grid_food_local/max_food]
                file.close()      
                if a_nb>0:nb_sim[i]+=1
        avg_grid_food,std_grid_food = [],[]
        a_avg_nb_agents,a_std_nb_agents = [],[]
        a_avg_food,a_std_food  = [],[]
        a_avg_min_food,a_std_min_food = [],[]
        a_avg_nb_offspring,a_std_nb_offspring = [],[]
        a_avg_food_trans,a_std_food_trans = [],[]
        a_avg_mut_prob,a_std_mut_prob = [],[]
        a_avg_generation,a_std_generation = [],[]
        #predators:
        p_avg_nb_agents,p_std_nb_agents = [],[]
        p_avg_food,p_std_food  = [],[]
        p_avg_min_food,p_std_min_food = [],[]
        p_avg_nb_offspring,p_std_nb_offspring = [],[]
        p_avg_food_trans,p_std_food_trans = [],[]
        p_avg_mut_prob,p_std_mut_prob = [],[]
        p_avg_generation,p_std_generation = [],[]
        
        for i in range(durationmax//step):
            if (10*i%100)==0:
                print(round(step*100*i/duration,2))
            avg_grid_food += [np.average(grid_food[i])]
            a_avg_nb_agents += [np.average(a_nb_agents[i])]
            a_avg_food += [np.average(a_food[i])]
            a_avg_min_food += [np.average(a_min_food[i])]
            a_avg_nb_offspring += [np.average(a_nb_offspring[i])]
            a_avg_food_trans += [np.average(a_food_trans[i])]
            a_avg_mut_prob += [np.average(a_mut_prob[i])]
            a_avg_generation += [np.average(a_generation[i])]
            #predators:
            p_avg_nb_agents += [np.average(p_nb_agents[i])]
            p_avg_food += [np.average(p_food[i])]
            p_avg_min_food += [np.average(p_min_food[i])]
            p_avg_nb_offspring += [np.average(p_nb_offspring[i])]
            p_avg_food_trans += [np.average(p_food_trans[i])]
            p_avg_mut_prob += [np.average(p_mut_prob[i])]
            p_avg_generation += [np.average(p_generation[i])]
            
            
            std_grid_food += [np.std(grid_food[i])*(nb_sim[i])**(-1/2)]
            a_std_nb_agents += [np.std(a_nb_agents[i])*(nb_sim[i])**(-1/2)]
            a_std_food += [np.std(a_food[i])*(a_avg_nb_agents[i])**(-1/2)]
            a_std_min_food += [np.std(a_min_food[i])*(a_avg_nb_agents[i])**(-1/2)]
            a_std_nb_offspring += [np.std(a_nb_offspring[i])*(a_avg_nb_agents[i])**(-1/2)]
            a_std_food_trans += [np.std(a_food_trans[i])*(a_avg_nb_agents[i])**(-1/2)]
            a_std_mut_prob += [np.std(a_mut_prob[i])*(a_avg_nb_agents[i])**(-1/2)]
            a_std_generation += [np.std(a_generation[i])*(a_avg_nb_agents[i])**(-1/2)]
            #predators:
            p_std_nb_agents += [np.std(p_nb_agents[i])*(nb_sim[i])**(-1/2)]
            p_std_food += [np.std(p_food[i])*(p_avg_nb_agents[i])**(-1/2)]
            p_std_min_food += [np.std(p_min_food[i])*(p_avg_nb_agents[i])**(-1/2)]
            p_std_nb_offspring += [np.std(p_nb_offspring[i])*(p_avg_nb_agents[i])**(-1/2)]
            p_std_food_trans += [np.std(p_food_trans[i])*(p_avg_nb_agents[i])**(-1/2)]
            p_std_mut_prob += [np.std(p_mut_prob[i])*(p_avg_nb_agents[i])**(-1/2)]
            p_std_generation += [np.std(p_generation[i])*(p_avg_nb_agents[i])**(-1/2)]
        
        with open('data/simulations_'+str(smin)+"_"+str(smax)+'.pickle', 'wb') as f:
            pickle.dump([T,avg_grid_food,a_avg_nb_agents,a_avg_food,a_avg_min_food,a_avg_nb_offspring,a_avg_food_trans,a_avg_mut_prob,a_avg_generation,p_avg_nb_agents,p_avg_food,p_avg_min_food,p_avg_nb_offspring,p_avg_food_trans,p_avg_mut_prob,p_avg_generation,std_grid_food,a_std_nb_agents,a_std_food,a_std_min_food,a_std_nb_offspring,a_std_food_trans,a_std_mut_prob,a_std_generation,p_std_nb_agents,p_std_food,p_std_min_food,p_std_nb_offspring,p_std_food_trans,p_std_mut_prob,p_std_generation], f, pickle.HIGHEST_PROTOCOL)
        with open('data/raw_simulations_'+str(smin)+"_"+str(smax)+'.pickle', 'wb') as f:
            pickle.dump([T,nb_sim,grid_food,a_nb_agents,a_food,a_min_food,a_nb_offspring,a_food_trans,a_mut_prob,a_generation,p_nb_agents,p_food,p_min_food,p_nb_offspring,p_food_trans,p_mut_prob,p_generation], f, pickle.HIGHEST_PROTOCOL)
            
    ## Histograms

    for i in range(len(T)):
        a_avg_nb_agents[i]*=mult
        a_std_nb_agents[i]*=mult        
        p_avg_nb_agents[i]*=mult
        p_std_nb_agents[i]*=mult
    
    predators=False
    # print(len(p_nb_agents[0]))
    # A=np.asarray(p_nb_agents)
    # print(np.shape(A))
    if p_nb_agents[0][0]!=0:predators=True
    
    
    if hist==1:
        
        if animate==1:
            
            data=a_food
            # exec("data="+variable+".copy()")
            durationmax=0
            for sim in range(smin,smax):
                file = open("data/simulation_"+str(sim)+"/datafile.txt","r")
                for line in file:
                    dat = line.split(" ")
                    if dat[0]=="duration":duration=int(dat[1])
                    if dat[0]=="step":step=int(dat[1])
                    if dat[0]=="max_food":max_food=float(dat[1])
                    if dat[0]=="n":n=int(dat[1])
                file.close()
                if duration>durationmax:simax=sim
            n, bins = np.histogram(data[-1], 100)
            # get the corners of the rectangles for the histogram
            left = np.array(bins[:-1])
            right = np.array(bins[1:])
            bottom = np.zeros(len(left))
            top = bottom + n*1.5
            nrects = len(left)
            
            nverts = nrects * (1 + 3 + 1)
            verts = np.zeros((nverts, 2))
            codes = np.ones(nverts, int) * path.Path.LINETO
            codes[0::5] = path.Path.MOVETO
            codes[4::5] = path.Path.CLOSEPOLY
            verts[0::5, 0] = left
            verts[0::5, 1] = bottom
            verts[1::5, 0] = left
            verts[1::5, 1] = top
            verts[2::5, 0] = right
            verts[2::5, 1] = top
            verts[3::5, 0] = right
            verts[3::5, 1] = bottom
            
            def animate(i):
                # simulate new data coming in
                t.sleep(0.1)
                nonlocal data
                time=50*i*step
                print(time)
                plot = data[50*i]
                n, bins,patches = plt.hist(plot, 100,log=True)
                top = bottom + n
                verts[1::5, 1] = top
                verts[2::5, 1] = top
                return [patch,]
            
            fig, ax = plt.subplots()
            barpath = path.Path(verts, codes)
            patch = patches.PathPatch(
                barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
            ax.add_patch(patch)
            
            ax.set_xlim(left[0], right[-1])
            ax.set_ylim(bottom.min(), top.max())
            
            ani = animation.FuncAnimation(fig, animate, int(duration/500), repeat=False, blit=True)
            plt.show()

        logscale=True
        plt.title("Histograms")
        
        # Setting log scale
        #1
        if predators : 
            hist,bins1,_=plt.hist(p_food[-1],bins=200,alpha=0.8,label="predator food",log=logscale)
        else:hist,bins1,_=plt.hist(a_food[-1],bins=200,alpha=0.8,label="predator food",log=logscale)
        logbins1 = np.logspace(np.log10(bins1[0]),np.log10(bins1[-1]),len(bins1))
        #2 
        hist,bins2,_=plt.hist(a_min_food[-1],bins=100,alpha=0.8,label="predator food",log=logscale)
        logbins2 = np.logspace(np.log10(bins2[0]),np.log10(bins2[-1]),len(bins2))
        #3
        hist,bins3,_=plt.hist(a_nb_offspring[-1],bins=100,alpha=0.8,label="predator food",log=logscale)
        logbins3 = np.logspace(np.log10(bins3[0]),np.log10(bins3[-1]),len(bins3))
        #4
        hist,bins4,_=plt.hist(a_food_trans[-1],bins=100,alpha=0.8,label="predator food",log=logscale)
        logbins4 = np.logspace(np.log10(bins4[0]),np.log10(bins4[-1]),len(bins4))
        plt.clf()
        #Plotting
        
        plt.subplot(2, 2, 1)
        plt.hist(a_food[-1],bins=logbins1,alpha=0.8,label="agent food",log=logscale)
        if predators : plt.hist(p_food[-1],bins=logbins1,alpha=0.8,label="predator food",log=logscale)
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.subplot(2, 2, 2)
        plt.hist(a_min_food[-1],bins=100,alpha=0.8,label="agent minfood",log=logscale)
        if predators : plt.hist(p_min_food[-1],bins=100,alpha=0.8,label="predator minfood",log=logscale)
        plt.legend()
        plt.tight_layout()
        # plt.xscale('log')
        
        plt.subplot(2, 2, 3)
        plt.hist(a_nb_offspring[-1],bins=100,alpha=0.8,label="agent nb_offspring",log=logscale)
        if predators : plt.hist(p_nb_offspring[-1],bins=100,alpha=0.8,label="predator nb_offspring",log=logscale)
        plt.legend()
        plt.tight_layout()
        # plt.xscale('log')
        
        plt.subplot(2, 2, 4)   
        plt.hist(a_food_trans[-1],bins=logbins4,alpha=0.8,label="agent food_trans",log=logscale)
        if predators : plt.hist(p_food_trans[-1],bins=logbins4,alpha=0.8,label="predator food_trans",log=logscale)
        plt.legend()
        plt.tight_layout()
        plt.xscale('log')
        plt.show()
        
        
        plt.hist(a_mut_prob[-1],bins=50,alpha=0.8,label="agent mut_prob",log=False)
        if predators : plt.hist(p_mut_prob[-1],bins=50,alpha=0.8,label="predator mut_prob",log=False)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.hist(a_generation[-1],bins=50,alpha=0.8,label="agent generation",log=False)
        if predators : plt.hist(p_generation[-1],bins=50,alpha=0.8,label="predator generation",log=False)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    #2d hist:
    if hist2d==1:
        plt.hist2d(a_min_food[-1], a_food_trans[-1], bins=(50,50), norm=colors.LogNorm(),range=[[0,5000],[0,500]])
        plt.xlabel("Agent min_food")
        plt.ylabel("Agent trans_food")
        plt.show()
        if predators : 
            plt.hist2d(p_min_food[-1], p_food_trans[-1], bins=(30,30), norm=colors.LogNorm())
            plt.xlabel("Predator min_food")
            plt.ylabel("Predator trans_food")
            plt.show()
        
        plt.hist2d(a_min_food[-1], a_food[-1], bins=(100,100), norm=colors.LogNorm())
        plt.xlabel("Agent min_food")
        plt.ylabel("Agent food")
        plt.show()
        if predators : 
            plt.hist2d(p_min_food[-1], p_food[-1], bins=(50,50), norm=colors.LogNorm(),label="predators",range=[[0,3000],[0,10000]])
            plt.xlabel("Predator min_food")
            plt.ylabel("Predator food")
            plt.show()
    
    
    ## Plots
    if plot==1:
        mut=1
        if a_avg_mut_prob[0]==0:mut=0
        
        #plot 0: nb of running simulations.
        plt.errorbar(T,nb_sim,label="Number of running simulations")
        plt.xlabel("day")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        #plot 1: mutation probability
        if mut==1:
            plt.errorbar(T,a_avg_mut_prob,yerr=a_std_mut_prob,elinewidth =0.3,label="Agent probability")
            if predators : plt.errorbar(T,p_avg_mut_prob,yerr=p_std_mut_prob,elinewidth =0.3,label="Predator probability")
            plt.xlabel("day")
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            plt.errorbar(T,a_avg_generation,yerr=a_std_generation,elinewidth =0.3,label="Agent generation")
            if predators : plt.errorbar(T,p_avg_generation,yerr=p_std_generation,elinewidth =0.3,label="Predator generation")
            plt.xlabel("day")
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        #plot 2: population
        
        plt.errorbar(T,avg_grid_food,yerr=std_grid_food,elinewidth =1,label="Food on the grid",color='g')
        plt.errorbar(T,a_avg_nb_agents,yerr=a_std_nb_agents,elinewidth =1,label=str(mult) +" * number of agents")

        if predators : plt.errorbar(T,p_avg_nb_agents,yerr=p_std_nb_agents,elinewidth =1,label=str(mult) +" * number of predators")
        plt.xlabel("day")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        #plot 3: genes
        genelog=0
        if mut==1:
            plt.title("Internal values")
            plt.subplot(2, 2, 1)
            # plt.title("avg_food")
            plt.xlabel("day")
            # plt.ylabel("average food stored")
            plt.errorbar(T,a_avg_food,yerr=a_std_food,elinewidth =0.3,label="Agents food stored ")
            if predators : plt.errorbar(T,p_avg_food,yerr=p_std_food,elinewidth =0.3,label="Predators food stored")
            plt.tight_layout()
            plt.legend()
            
            
            if genelog:plt.yscale("log")
            
            plt.subplot(2, 2, 2)
            # plt.title("avg_min_food")
            plt.xlabel("day")
            # plt.ylabel("average min_food")
            plt.errorbar(T,a_avg_min_food,yerr=a_std_min_food,elinewidth =0.25,label="Agents min_food")
            if predators : plt.errorbar(T,p_avg_min_food,yerr=p_std_min_food,elinewidth =0.25,label="Predators min_food")
            plt.tight_layout() 
            plt.legend()
            
            plt.subplot(2, 2, 3)
            # plt.title("avg_nb_offspring")
            plt.xlabel("day")
            # plt.ylabel("average nb_offspring")
            plt.errorbar(T,a_avg_nb_offspring,yerr=a_std_nb_offspring,elinewidth =0.25,label="Agents nb_offspring")
            if predators : plt.errorbar(T,p_avg_nb_offspring,yerr=p_std_nb_offspring,elinewidth =0.25,label="Predators nb_offspring")
            plt.tight_layout() 
            plt.legend(loc=3)
            
            plt.subplot(2, 2, 4)
            # plt.title("avg_food_trans")
            plt.xlabel("day")
            # plt.ylabel("average food_trans")
            
            plt.errorbar(T,a_avg_food_trans,yerr=a_std_food_trans,elinewidth =0.25,label="Agents food_trans")
            if predators : plt.errorbar(T,p_avg_food_trans,yerr=p_std_food_trans,elinewidth =0.25,label="Predators food_trans")
            plt.tight_layout() 
            plt.legend()
            if genelog:plt.yscale("log")
            plt.show()
        else:
            plt.xlabel("day")
            # plt.ylabel("average food stored")
            plt.errorbar(T,a_avg_food,yerr=a_std_food,elinewidth =0.25,label="Agents food stored ")
            if predators : plt.errorbar(T,p_avg_food,yerr=p_std_food,elinewidth =0.25,label="Predators food stored")
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        # plt.show()

    return T,nb_sim,grid_food,a_nb_agents,a_food,a_min_food,a_nb_offspring,a_food_trans,a_mut_prob,a_generation,p_nb_agents,p_food,p_min_food,p_nb_offspring,p_food_trans,p_mut_prob,p_generation


def comparison(s0,s1,s2,s3,label1,label2):
    T,avg_grid_food,a_avg_nb_agents,a_avg_food,a_avg_min_food,a_avg_nb_offspring,a_avg_food_trans,a_avg_mut_prob,a_avg_generation,p_avg_nb_agents,p_avg_food,p_avg_min_food,p_avg_nb_offspring,p_avg_food_trans,p_avg_mut_prob,p_avg_generation,std_grid_food,a_std_nb_agents,a_std_food,a_std_min_food,a_std_nb_offspring,a_std_food_trans,a_std_mut_prob,a_std_generation,p_std_nb_agents,p_std_food,p_std_min_food,p_std_nb_offspring,p_std_food_trans,p_std_mut_prob,p_std_generation = pickle.load(open('simulations_'+str(s0)+"_"+str(s1)+'.pickle', "rb" ))
    T,avg_grid_food2,a_avg_nb_agents2,a_avg_food2,a_avg_min_food2,a_avg_nb_offspring2,a_avg_food_trans2,a_avg_mut_prob2,a_avg_generation2,p_avg_nb_agents2,p_avg_food2,p_avg_min_food2,p_avg_nb_offspring2,p_avg_food_trans2,p_avg_mut_prob2,p_avg_generation2,std_grid_food2,a_std_nb_agents2,a_std_food2,a_std_min_food2,a_std_nb_offspring2,a_std_food_trans2,a_std_mut_prob2,a_std_generation2,p_std_nb_agents2,p_std_food2,p_std_min_food2,p_std_nb_offspring2,p_std_food_trans2,p_std_mut_prob2,p_std_generation2 = pickle.load(open('simulations_'+str(s2)+"_"+str(s3)+'.pickle', "rb" ))
    
    predators=False
    # print(len(p_nb_agents[0]))
    # A=np.asarray(p_nb_agents)
    # print(np.shape(A))
    # if p_nb_agents[0][0]!=0:predators=True
    
    mut=1
    if a_avg_mut_prob[0]==0:mut=0
        #plot 1: mutation probability
    if mut==1:
        plt.errorbar(T,a_avg_mut_prob,yerr=a_std_mut_prob,elinewidth =0.25,label="Agent probability " + label1)
        plt.errorbar(T,a_avg_mut_prob2,yerr=a_std_mut_prob2,elinewidth =0.25,label="Agent probability " + label2)
        if predators : 
            plt.errorbar(T,p_avg_mut_prob,yerr=p_std_mut_prob,elinewidth =0.25,label="Predator probability " + label1)
            plt.errorbar(T,p_avg_mut_prob2,yerr=p_std_mut_prob2,elinewidth =0.25,label="Predator probability " + label2)
        plt.xlabel("day")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        plt.errorbar(T,a_avg_generation,yerr=a_std_generation,elinewidth =0.25,label="Agent generation " + label1)
        plt.errorbar(T,a_avg_generation2,yerr=a_std_generation2,elinewidth =0.25,label="Agent generation " + label2)
        if predators :
            plt.errorbar(T,p_avg_generation,yerr=p_std_generation,elinewidth =0.25,label="Predator generation " + label1)
            plt.errorbar(T,p_avg_generation2,yerr=p_std_generation2,elinewidth =0.25,label="Predator generation " + label2)
        plt.xlabel("day")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    #plot 2: population
    
    plt.errorbar(T,avg_grid_food,yerr=std_grid_food,elinewidth =1,label="Food on the grid")
    plt.errorbar(T,a_avg_nb_agents,yerr=a_std_nb_agents,elinewidth =1,label=str(mult) +" * number of agents")
    
    plt.errorbar(T,avg_grid_food2,yerr=std_grid_food2,elinewidth =1,label="Food on the grid " + label2)
    plt.errorbar(T,a_avg_nb_agents2,yerr=a_std_nb_agents2,elinewidth =1,label=str(mult) +" * number of agents " + label2)
    if predators :
        plt.errorbar(T,p_avg_nb_agents,yerr=p_std_nb_agents,elinewidth =1,label=str(mult) +" * number of predators")
        plt.errorbar(T,p_avg_nb_agents2,yerr=p_std_nb_agents2,elinewidth =1,label=str(mult) +" * number of predators " + label2)
    plt.xlabel("day")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    #plot 3: genes
    genelog=0
    if mut==1:
        plt.title("Internal values")
        plt.subplot(2, 2, 1)
        # plt.title("avg_food")
        plt.xlabel("day")
        # plt.ylabel("average food stored")
        plt.errorbar(T,a_avg_food,yerr=a_std_food,elinewidth =0.25,label="a_food " + label1)
        if predators : plt.errorbar(T,p_avg_food,yerr=p_std_food,elinewidth =0.25,label="p_food " + label1)
        plt.errorbar(T,a_avg_food2,yerr=a_std_food2,elinewidth =0.25,label="a_food seasons ")
        if predators : plt.errorbar(T,p_avg_food2,yerr=p_std_food2,elinewidth =0.25,label="p_food " + label2)
        plt.tight_layout()
        plt.legend()
        if genelog:plt.yscale("log")
        
        plt.subplot(2, 2, 2)
        # plt.title("avg_min_food")
        plt.xlabel("day")
        # plt.ylabel("average min_food")
        plt.errorbar(T,a_avg_min_food,yerr=a_std_min_food,elinewidth =0.25,label="a_min_food " + label1)
        if predators : plt.errorbar(T,p_avg_min_food,yerr=p_std_min_food,elinewidth =0.25,label="p_min_food " + label1)
        plt.errorbar(T,a_avg_min_food2,yerr=a_std_min_food2,elinewidth =0.25,label="a_min_food " + label2)
        if predators : plt.errorbar(T,p_avg_min_food2,yerr=p_std_min_food2,elinewidth =0.25,label="p_min_food " + label2)
        plt.tight_layout()
        plt.legend()
        plt.subplot(2, 2, 3)
        # plt.title("avg_nb_offspring")
        plt.xlabel("day")
        # plt.ylabel("average nb_offspring")
        plt.errorbar(T,a_avg_nb_offspring,yerr=a_std_nb_offspring,elinewidth =0.25,label="a_nb_offspring " + label1)
        if predators : plt.errorbar(T,p_avg_nb_offspring,yerr=p_std_nb_offspring,elinewidth =0.25,label="p_nb_offspring " + label1)
        plt.tight_layout()       
        plt.errorbar(T,a_avg_nb_offspring2,yerr=a_std_nb_offspring2,elinewidth =0.25,label="a_nb_offspring " + label2)
        if predators : plt.errorbar(T,p_avg_nb_offspring2,yerr=p_std_nb_offspring2,elinewidth =0.25,label="p_nb_offspring " + label2)
        plt.tight_layout()
        plt.legend()
        plt.subplot(2, 2, 4)
        # plt.title("avg_food_trans")
        plt.xlabel("day")
        # plt.ylabel("average food_trans")
        
        plt.errorbar(T,a_avg_food_trans,yerr=a_std_food_trans,elinewidth =0.25,label="a_food_trans " + label1)
        if predators : plt.errorbar(T,p_avg_food_trans,yerr=p_std_food_trans,elinewidth =0.25,label="p_food_trans " + label1)
        
        plt.errorbar(T,a_avg_food_trans2,yerr=a_std_food_trans2,elinewidth =0.25,label="a_food_trans " + label2)
        if predators : plt.errorbar(T,p_avg_food_trans2,yerr=p_std_food_trans2,elinewidth =0.25,label="p_food_trans " + label2)
        plt.tight_layout()
        plt.legend()
        if genelog:plt.yscale("log")
        plt.show()
    else:
        plt.xlabel("day")
        # plt.ylabel("average food stored")
        plt.errorbar(T,a_avg_food,yerr=a_std_food,elinewidth =0.25,label="Agents food stored ")
        if predators : plt.errorbar(T,p_avg_food,yerr=p_std_food,elinewidth =0.25,label="Predators food stored")
        plt.errorbar(T,a_avg_food2,yerr=a_std_food2,elinewidth =0.25,label="Agents food stored2 ")
        if predators : plt.errorbar(T,p_avg_food2,yerr=p_std_food2,elinewidth =0.25,label="Predators food stored2")
        plt.legend()
        plt.tight_layout()
        plt.show()
    

    


##execute

smin,smax = 6, 10

parameters=["plot","hist"] #"plot" outputs the plots, "hist" the histograms, "override" doesn't load the previously saved data and computes the raw data again, "animate" outputs a histogram animation (doesn't work quite well)

# T,nb_sim,grid_food,a_nb_agents,a_food,a_min_food,a_nb_offspring,a_food_trans,a_mut_prob,a_generation,p_nb_agents,p_food,p_min_food,p_nb_offspring,p_food_trans,p_mut_prob,p_generation=weighted_average(smin,smax,parameters)

weighted_average(smin, smax)