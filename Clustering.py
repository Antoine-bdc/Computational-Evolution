import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import os as os
import pickle 
import numpy.fft as fft
import time 
from matplotlib.font_manager import FontProperties
from matplotlib import colors
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
import pylab as pl
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from itertools import cycle
from sklearn.cluster import AffinityPropagation

os.chdir("D:\\Documents\\Etudes\\Spore\\reproduction\\data ")
dir=os.getcwd()

def read(smin,smax,day,parameters=["food","min_food","food_trans"]):
    T,nb_sim,grid_food,a_nb_agents,a_food,a_min_food,a_nb_offspring,a_food_trans,a_mut_prob,a_generation,p_nb_agents,p_food,p_min_food,p_nb_offspring,p_food_trans,p_mut_prob,p_generation = pickle.load(open('raw_simulations_'+str(smin)+"_"+str(smax)+'.pickle', "rb" ))
    print("Loading data")
    
    a_data=[]
    p_data=[]
    if "food" in parameters:
        a_data.append(a_food[day])
        p_data.append(p_food[day])
    if "min_food" in parameters:
        a_data.append(a_min_food[day])
        p_data.append(p_min_food[day])
    if "food_trans" in parameters:
        a_data.append(a_food_trans[day])
        p_data.append(p_food_trans[day])
    if "nb_offspring" in parameters:
        a_data.append(a_nb_offspring[day])
        p_data.append(p_nb_offspring[day])
    if "probability" in parameters:
        a_data.append(a_mut_prob[day])
        p_data.append(p_mut_prob[day])
    a_data=np.transpose(np.asarray(a_data))
    p_data=np.transpose(np.asarray(p_data))
    print(len(a_data))
    print(len(p_data))
    print("min",np.min(p_data))
        
    return(a_data,p_data)

def sorting_data(parameters,a_data,p_data,a_n_clusters_,p_n_clusters_,a_labels,p_labels):
    count=0
    ALL=[]
    if "food" in parameters:
        a_food=[]
        for k in range(a_n_clusters_):
            a_food.append([])
        for k in range(a_n_clusters_):
            for i in range(len(a_data)):
                if k==a_labels[i]:
                    a_food[k].append(a_data[i,count])
        p_food=[]
        for k in range(p_n_clusters_):
            p_food.append([])
        for k in range(p_n_clusters_):
            for i in range(len(p_data)):
                if k==p_labels[i]:
                    p_food[k].append(p_data[i,count])
        count+=1
        ALL.append(a_food)
        ALL.append(p_food)
    else:
        ALL.append(None)
        ALL.append(None)
    if "min_food" in parameters:
        a_min_food=[]
        for k in range(a_n_clusters_):
            a_min_food.append([])
        for k in range(a_n_clusters_):
            for i in range(len(a_data)):
                if k==a_labels[i]:
                    a_min_food[k].append(a_data[i,count])
        p_min_food=[]
        for k in range(p_n_clusters_):
            p_min_food.append([])
        for k in range(p_n_clusters_):
            for i in range(len(p_data)):
                if k==p_labels[i]:
                    p_min_food[k].append(p_data[i,count])
        count+=1
        ALL.append(a_min_food)
        ALL.append(p_min_food)
    else:
        ALL.append(None)
        ALL.append(None)
    if "food_trans" in parameters:
        a_food_trans=[]
        for k in range(a_n_clusters_):
            a_food_trans.append([])
        for k in range(a_n_clusters_):
            for i in range(len(a_data)):
                if k==a_labels[i]:
                    a_food_trans[k].append(a_data[i,count])
        p_food_trans=[]
        for k in range(p_n_clusters_):
            p_food_trans.append([])
        for k in range(p_n_clusters_):
            for i in range(len(p_data)):
                if k==p_labels[i]:
                    p_food_trans[k].append(p_data[i,count])
        count+=1
        ALL.append(a_food_trans)
        ALL.append(p_food_trans)
    else:
        ALL.append(None)
        ALL.append(None)
         
    if "nb_offspring" in parameters:
        a_nb_offspring=[]
        for k in range(a_n_clusters_):
            a_nb_offspring.append([])
        for k in range(a_n_clusters_):
            for i in range(len(a_data)):
                if k==a_labels[i]:
                    a_nb_offspring[k].append(a_data[i,count])
        p_nb_offspring=[]
        for k in range(p_n_clusters_):
            p_nb_offspring.append([])
        for k in range(p_n_clusters_):
            for i in range(len(p_data)):
                if k==p_labels[i]:
                    p_nb_offspring[k].append(p_data[i,count])
        count+=1
        ALL.append(a_nb_offspring)
        ALL.append(p_nb_offspring)
    else:
        ALL.append(None)
        ALL.append(None)
        
    if "probability" in parameters:
        a_probability=[]
        for k in range(a_n_clusters_):
            a_probability.append([])
        for k in range(a_n_clusters_):
            for i in range(len(a_data)):
                if k==a_labels[i]:
                    a_probability[k].append(a_data[i,count])
        p_probability=[]
        for k in range(p_n_clusters_):
            p_probability.append([])
        for k in range(p_n_clusters_):
            for i in range(len(p_data)):
                if k==p_labels[i]:
                    p_probability[k].append(p_data[i,count])
        count+=1
        ALL.append(a_probability)
        ALL.append(p_probability)
    else:
        ALL.append(None)
        ALL.append(None)
        
    return(ALL)

# Compute clustering with MeanShift
def meanshift_clustering(a_data,p_data):
    # The following bandwidth can be automatically detected using
    a_bandwidth = estimate_bandwidth(a_data, quantile=0.2, n_samples=500)
    p_bandwidth = estimate_bandwidth(p_data, quantile=0.2, n_samples=500)
    
    
    a_bandwidth*=2.2
    p_bandwidth*=1
    
    print(a_bandwidth)
    print(p_bandwidth)
    
    a_ms = MeanShift(bandwidth=a_bandwidth, bin_seeding=True)
    
    a_ms.fit(a_data)
    a_labels = a_ms.labels_
    a_cluster_centers = a_ms.cluster_centers_
    p_ms = MeanShift(bandwidth=p_bandwidth, bin_seeding=True)
    p_ms.fit(p_data)
    p_labels = p_ms.labels_
    p_cluster_centers = p_ms.cluster_centers_
    
    a_labels_unique = np.unique(a_labels)
    a_n_clusters_ = len(a_labels_unique)
    p_labels_unique = np.unique(p_labels)
    p_n_clusters_ = len(p_labels_unique)
    
    # for i in range(len(a_labels)):
    #     temp=a_labels[i]
    #     if temp==0:a_labels[i]=2
    #     if temp==2:a_labels[i]=0
    
    print("number of a_clusters : %d" % a_n_clusters_)
    print("number of p_clusters : %d" % p_n_clusters_)
    
    #sorting data
    ALL=sorting_data(parameters,a_data,p_data,a_n_clusters_,p_n_clusters_,a_labels,p_labels)
    return(ALL,parameters)

def dbscan_clustering(a_data,p_data,eps,min_sample):
    atot=len(a_data)
    ptot=len(p_data)
    a_db = DBSCAN(eps=eps[0], min_samples=min_sample[0]).fit(a_data)
    a_core_samples_mask = np.zeros_like(a_db.labels_, dtype=bool)
    a_core_samples_mask[a_db.core_sample_indices_] = True
    a_labels = a_db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    a_n_clusters_= len(set(a_labels)) - (1 if -1 in a_labels else 0)
    a_n_noise_ = list(a_labels).count(-1)
    
    print('Estimated number of a_clusters: %d' % a_n_clusters_)
    print('Estimated number of a_noise points:',a_n_noise_, "(",round(100*a_n_noise_/atot,2),"%)")
    
    
    p_db = DBSCAN(eps=eps[1], min_samples=min_sample[1]).fit(p_data)
    p_core_samples_mask = np.zeros_like(p_db.labels_, dtype=bool)
    p_core_samples_mask[p_db.core_sample_indices_] = True
    p_labels = p_db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    p_n_clusters_= len(set(p_labels)) - (1 if -1 in p_labels else 0)
    p_n_noise_ = list(p_labels).count(-1)
    
    print('Estimated number of p_clusters: %d' % p_n_clusters_)
    print('Estimated number of a_noise points:',p_n_noise_, "(",round(100*p_n_noise_/ptot,2),"%)")
    
    a_unique_labels = set(a_labels)
    a_colors = [plt.cm.Spectral(each)
            for each in np.linspace(0, 1, len(a_unique_labels))]
    for k, col in zip(a_unique_labels, a_colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 0.7]
    
    #sorting data
    ALL=sorting_data(parameters,a_data,p_data,a_n_clusters_,p_n_clusters_,a_labels,p_labels)
    a_noise,p_noise=0,0
    if a_n_noise_>0:a_noise=True
    if p_n_noise_>0:p_noise=True
    return(ALL,parameters,a_noise,p_noise)



# Compute clustering with MeanShift
def affinity_propagation_clustering(a_data,p_data,parameters):
    # The following bandwidth can be automatically detected using

    a_af = AffinityPropagation(preference=-3e6,damping=0.9999).fit(a_data)
    a_cluster_centers_indices = a_af.cluster_centers_indices_
    a_labels = a_af.labels_
    a_n_clusters_ = len(a_cluster_centers_indices)
    
    
    p_af = AffinityPropagation(preference=-5e5,damping=0.9999).fit(p_data)
    p_cluster_centers_indices = p_af.cluster_centers_indices_
    p_labels = p_af.labels_
    p_n_clusters_ = len(p_cluster_centers_indices)
    

    print("Agents Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(a_data, a_labels, metric='sqeuclidean'))
    print("Predators Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(p_data, p_labels, metric='sqeuclidean'))
    
    print("number of a_clusters : %d" % a_n_clusters_)
    print("number of p_clusters : %d" % p_n_clusters_)
    
    #sorting data
    ALL=sorting_data(parameters,a_data,p_data,a_n_clusters_,p_n_clusters_,a_labels,p_labels)
    
    return(ALL,parameters)

    
    
def plot_clusters(ALL,parameters,smin,smax,day,abins,pbins,a_noise=0,p_noise=0):
    
    T,nb_sim,grid_food,a_nb_agents,a_food,a_min_food,a_nb_offspring,a_food_trans,a_mut_prob,a_generation,p_nb_agents,p_food,p_min_food,p_nb_offspring,p_food_trans,p_mut_prob,p_generation = pickle.load(open('raw_simulations_'+str(smin)+"_"+str(smax)+'.pickle', "rb" ))
    
    count=0
    if "food" in parameters:
        a_food=ALL[0]
        p_food=ALL[1]
        count+=1
    else:
        a_food=a_food[day]
        p_food=p_food[day]
        
    if "min_food" in parameters:
        a_min_food=ALL[2]
        p_min_food=ALL[3]
        count+=1
    else:
        a_min_food=a_min_food[day]
        p_min_food=p_min_food[day]
        
    if "food_trans" in parameters:
        a_food_trans=ALL[4]
        p_food_trans=ALL[5]
        count+=1
    else:
        a_food_trans=a_food_trans[day]
        p_food_trans=p_food_trans[day]
        
    if "nb_offspring" in parameters:
        a_nb_offspring=ALL[6]
        p_nb_offspring=ALL[7]
        count+=1
    else:
        a_nb_offspring=a_nb_offspring[day]
        p_nb_offspring=p_nb_offspring[day]
        
    if "probability" in parameters:
        a_probability=ALL[8]
        p_probability=ALL[9]
        count+=1
    else:
        a_probability=a_mut_prob[day]
        p_probability=p_mut_prob[day]

    # Setting log scale
    logscale=True
    #1
    hist,bins1,_=plt.hist(p_food,bins=abins[0],alpha=0.8,label="predator food",log=logscale, density=False, histtype='bar', stacked=True)
    logbins1 = np.logspace(np.log10(bins1[0]),np.log10(bins1[-1]),len(bins1))
    print(-1)
    #2 
    hist,bins2,_=plt.hist(a_min_food,bins=abins[1],alpha=0.8,label="predator food",log=logscale, density=False, histtype='bar', stacked=True)
    logbins2 = np.logspace(np.log10(bins2[0]),np.log10(bins2[-1]),len(bins2))
    #4
    print(0)
    hist,bins4,_=plt.hist(a_food_trans,bins=abins[2],alpha=0.8,label="predator food",log=logscale, density=False, histtype='bar', stacked=True)
    logbins4 = np.logspace(np.log10(bins4[0]),np.log10(bins4[-1]),len(bins4))
    #3
    print(1)
    """
    hist,bins3,_=plt.hist(a_nb_offspring,bins=100,alpha=0.8,label="predator food",log=logscale, density=False, histtype='bar', stacked=True)
    logbins3 = np.logspace(np.log10(bins3[0]),np.log10(bins3[-1]),len(bins3))
    #5
    print(2)
    hist,bins5,_=plt.hist(a_probability,bins=100,alpha=0.8,label="predator food",log=logscale, density=False, histtype='bar', stacked=True)
    logbins5 = np.logspace(np.log10(bins5[0]),np.log10(bins5[-1]),len(bins5))
    print(3)
    """
    #Plotting agents
    
    if a_noise>0:
        a_food_noise=a_food.pop(-1)
        a_food_trans_noise=a_food_trans.pop(-1)
        a_min_food_noise=a_min_food.pop(-1)
        a_nb_offspring_noise=a_nb_offspring.pop(-1)
        a_probability_noise=a_probability.pop(-1)
    
    # plt.subplot(3, 2, 1)
    # plt.clf()
    """
    plt.subplot(3, 2, 2)
    plt.hist(a_food,bins=logbins1,alpha=0.8,label="agent food",log=logscale, density=False, histtype='bar', stacked=True)
    if a_noise>0:
        plt.hist(a_food_noise,bins=logbins1,alpha=0.5,label="noise",log=logscale, density=False, histtype='bar', color="black")
    plt.xscale('log')
    plt.legend() 
    plt.tight_layout()
    """
    plt.subplot(2, 2, 1)
    plt.hist(a_min_food,bins=logbins2,alpha=0.8,label="agent minfood",log=logscale, density=False, histtype='bar', stacked=True)
    if a_noise>0:
        plt.hist(a_min_food_noise,bins=logbins2,alpha=0.5,label="noise",log=logscale, density=False, histtype='bar', color="black")
    plt.legend() 
    plt.tight_layout()
    plt.xscale('log')
    
        
    plt.subplot(2, 2, 2)   
    plt.hist(a_food_trans,bins=logbins4,alpha=0.8,label="agent food_trans",log=logscale, density=False, histtype='bar', stacked=True)
    if a_noise>0:
        plt.hist(a_food_trans_noise,bins=logbins4,alpha=0.5,label="noise",log=logscale, density=False, histtype='bar', color="black")
    plt.legend() 
    plt.tight_layout()
    plt.xscale('log')
    
    plt.subplot(2, 2, 3)   
    plt.hist(a_nb_offspring,bins=abins[3],alpha=0.8,label="agent nb offspring", density=False,log=logscale, histtype='bar', stacked=True)
    if a_noise>0:
        plt.hist(a_nb_offspring_noise,bins=abins[3],alpha=0.5,log=logscale,label="noise", density=False, histtype='bar', color="black")
    plt.legend() 
    plt.tight_layout()
    
    plt.subplot(2, 2, 4)   
    plt.hist(a_probability,bins=abins[4],alpha=0.8,label="agent probability", density=False,log=logscale, histtype='bar', stacked=True)
    if a_noise>0:
        plt.hist(a_probability_noise,bins=abins[4],alpha=0.5,log=logscale,label="noise",density=False, histtype='bar', color="black")
    
    plt.legend() 
    plt.tight_layout()
    plt.show()
    
    #plotting predators
    logscale=False
    
    if p_noise>0:
        p_food_noise=p_food.pop(-1)
        p_food_trans_noise=p_food_trans.pop(-1)
        p_min_food_noise=p_min_food.pop(-1)
        p_nb_offspring_noise=p_nb_offspring.pop(-1)
        p_probability_noise=p_probability.pop(-1)
    
    # plt.subplot(3, 2, 1)
    # plt.clf()
    """
    plt.subplot(3, 2, 2)
    plt.hist(p_food,bins=logbins1,alpha=0.8,label="predator food",log=logscale, density=False, histtype='bar', stacked=True)
    if p_noise>0:
        plt.hist(p_food_noise,bins=logbins1,alpha=0.5,label="noise",log=logscale, density=False, histtype='bar', color="black"
    plt.xscale('log')
    plt.legend() 
    plt.tight_layout()
    """
    plt.subplot(2, 2, 1) 
    plt.hist(p_min_food,bins=logbins2,alpha=0.8,label="predator minfood",log=logscale, density=False, histtype='bar', stacked=True)
    if p_noise>0:
        plt.hist(p_min_food_noise,bins=logbins2,alpha=0.5,label="noise",log=logscale, density=False, histtype='bar', color="black")
    plt.legend() 
    plt.tight_layout()
    plt.xscale('log')
    
    plt.subplot(2, 2, 2) 
    plt.hist(p_food_trans,bins=logbins4,alpha=0.8,label="predator food_trans",log=logscale, density=False, histtype='bar', stacked=True)
    if p_noise>0:
        plt.hist(p_food_trans_noise,bins=logbins2,alpha=0.5,label="noise",log=logscale, density=False, histtype='bar', color="black")
    plt.legend() 
    plt.tight_layout()
    plt.xscale("log")
    
    plt.subplot(2, 2, 3)    
    plt.hist(p_nb_offspring,bins=pbins[3],log=logscale,alpha=0.8,label="predator nb offspring", density=False, histtype='bar', stacked=True)
    if p_noise>0:
        plt.hist(p_nb_offspring_noise,bins=pbins[3],log=logscale,alpha=0.5,label="noise", density=False, histtype='bar', color="black")
    plt.legend() 
    plt.tight_layout()
    
    plt.subplot(2, 2, 4)   
    plt.hist(p_probability,bins=pbins[4],log=logscale,alpha=0.8,label="predator probability", density=False, histtype='bar', stacked=True)
    if p_noise>0:
        plt.hist(p_probability_noise,bins=pbins[4],log=logscale,alpha=0.5,label="noise",density=False, histtype='bar', color="black")
    
    plt.legend() 
    plt.tight_layout()
    plt.show()


### Execute 


parameters=["foofd","min_food","food_trans","nb_offspring","probability"]
smin,smax,day=100,120,-1
abinsize=[100,50,50,50,50]
pbinsize=[100,50,50,50,50]
n_a,n_p=0,0

try:
    plot_clusters(ALL,parameters,smin,smax,day,abinsize,pbinsize,n_a,n_p)

except:
    a_data,p_data=read(smin,smax,day,parameters)
    ALL,parameters=meanshift_clustering(a_data,p_data)
    plot_clusters(ALL,parameters,smin,smax,day,abinsize,pbinsize,n_a,n_p)


# eps=[600,600]
# min=[20,7]
# ALL,parameters,n_a,n_p=dbscan_clustering(a_data,p_data,eps,min)

# ALL,parameters=affinity_propagation_clustering(a_data,p_data,parameters)



