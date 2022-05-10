import numpy as np
import random
import time
import os


class Agent:
    def __init__(self,type, food, dying_threshold, minimum_food, position,nb_offspring=1,food_transmitted=100,generation=0,mutation_probability=0,multiplicator=1,age=0,totalchildren=0,name=0):
        """on initialise toutes les caracteristiques propres de lagent avec les donnees fournies en argument"""
        self.type = type # type=1 for agent and type=2 for predator
        self.food = food
        self.dying_threshold = dying_threshold
        self.minimum_food = minimum_food
        self.position = position
        self.nb_offspring=nb_offspring
        self.food_transmitted=food_transmitted
        self.generation=generation
        self.age=age
        self.totalchildren=totalchildren
        self.mutation_probability=mutation_probability
        self.multiplicator = multiplicator
        self.last_meal = 0.0
        """if name==0:
            self.name=namegen.generate_name(random.randint(2,5))
        else:self.name=name
        """

    def increase_food(self, food_eaten):
        self.food += food_eaten

    def check_evolution(self):
        if random.random()>self.nb_offspring-int(self.nb_offspring):
            number=int(self.nb_offspring)
        else:
            number=int(self.nb_offspring)+1
        if self.food-self.minimum_food - self.food_transmitted*number >=0 : #The cell will not let itself go under self.minimum_food when giving birth. minimum_food replaces birth_threshold
            return number
        if self.food <= self.dying_threshold :
            return -1
        return 0

    def change_position(self, grid,agent_list):
        moved=0
        old_position = self.position
        new_position = four_directions(grid,self.position)
        self.position = new_position
        self.eat_food(grid,agent_list)
        # if self.food<10:
            # print("survivor",self.type,self.food)
            # print(self.minimum_food)
            # print(self.food_transmitted)
            # print(self.mutation_probability)
        if self.type == 1:
            self.food-=10
        elif self.type == 2:
            self.food -=10
        # if self.food<0:print("survivor",self.type,self.food)
        grid[old_position[0]][old_position[1]][0]=False  #Update the grid
        grid[new_position[0]][new_position[1]][0]=self.type
        self.age+=1

    def eat_food(self,grid,agent_list):
        if self.type==1:
            self.increase_food(grid[self.position[0]][self.position[1]][1])
            grid[self.position[0]][self.position[1]][1]=0
        if self.type==2:
            neigh = neighbours(grid,self.position)
            if 1 in neigh:
                eaten=0
                tested = [0]*4
                while eaten==0:
                    d4=random.randint(0,3)
                    tested[d4] = 1
                    if neigh[d4]==1:
                        nx=len(grid)
                        ny=len(grid[0])
                        pos=self.position
                        if d4==0 : prey_position=[(pos[0]-1)%nx,(pos[1])%ny]
                        if d4==1 : prey_position=[(pos[0])%nx,(pos[1]+1)%ny]
                        if d4==2 : prey_position=[(pos[0]+1)%nx,(pos[1])%ny]
                        if d4==3 : prey_position=[(pos[0])%nx,(pos[1]-1)%ny]
                        eaten=1
                        for i in range(len(agent_list)):
                            if agent_list[i].position==prey_position and agent_list[i].food < self.food*self.multiplicator:
                                food_eaten=agent_list[i].food
                                agent_list.pop(i)
                                grid[prey_position[0]][prey_position[1]][0]=0
                                self.increase_food(food_eaten)
                                self.last_meal = food_eaten
                                break
                    if tested == [1]*4:
                        break




    def give_birth(self,grid,number):
        offspring=[]
        if number<=4-nb_neighbours(grid,self.position):
            for k in range (number):
                coord = four_directions(grid,self.position)
                grid[coord[0]][coord[1]][0]=self.type
                self.increase_food(-1*self.food_transmitted)
                offspring+=[Agent(self.type,self.food_transmitted,self.dying_threshold,max(self.minimum_food,mutate(self.minimum_food,self.mutation_probability)),coord,mutate(self.nb_offspring,self.mutation_probability),mutate(self.food_transmitted,self.mutation_probability),self.generation+1,min(self.mutation_probability,mutate(self.mutation_probability,self.mutation_probability)),multiplicator =mutate(self.multiplicator,0))]
                self.totalchildren+=number
        return offspring

def neighbours(grid,pos):
    nx=len(grid)
    ny=len(grid[0])
    return([grid[(pos[0]-1)%nx][(pos[1])%ny][0],grid[(pos[0])%nx][(pos[1]+1)%ny][0],grid[(pos[0]+1)%nx][(pos[1])%ny][0],grid[(pos[0])%nx][(pos[1]-1)%ny][0]])

def nb_neighbours(grid,pos):
    nx=len(grid)
    ny=len(grid[0])
    nei=neighbours(grid,pos)
    count=0
    if nei[0]!=0:count+=1 #up
    if nei[1]!=0:count+=1 #right
    if nei[2]!=0:count+=1 #down
    if nei[3]!=0:count+=1 #left
    return(count)

def four_directions(grid, old_position):
    moved=0
    nx,ny=len(grid),len(grid[0])
    pos=[old_position[0],old_position[1]]
    if nb_neighbours(grid,old_position)==4:
        return(old_position)
    else:
        while moved==0:
            d4=random.randint(1,4) # 1=up, 2=right, 3=down, 4=left
            if d4==1:
                if  grid[(pos[0]-1)%nx][(pos[1])%ny][0]==False:
                    pos[0]=(pos[0]-1)%len(grid)
                    moved=1
            if d4==2:
                if grid[(pos[0])%nx][(pos[1]+1)%ny][0]==False:
                    pos[1]=(pos[1]+1)%len(grid[0])
                    moved=1
            if d4==3:
                if grid[(pos[0]+1)%nx][(pos[1])%ny][0]==False:
                    pos[0]=(pos[0]+1)%len(grid)
                    moved=1
            if d4==4:
                if grid[(pos[0])%nx][(pos[1]-1)%ny][0]==False:
                    pos[1]=(pos[1]-1)%len(grid[0])
                    moved=1
        return(pos)

def mutate(value,probability):   # probability is between [0,1]
    if random.random()<probability:
        return(value+(random.random()-0.5)*value/2) # Uniform random mutation between 75% and 125% of initial value
    else :return(value)

def set_position(grid):
    """fonction qui retourne une liste [i,j] qui sera la position de la nourriture (ou de l'agent)"""
    x = random.randint(0,len(grid)-1)
    y = random.randint(0,len(grid[0])-1)
    while grid[x][y][0] !=0 :
         x = random.randint(0,len(grid)-1)
         y = random.randint(0,len(grid[0])-1)
    return [x,y]

def create_grid(n, nbAgents,nb_predators,max_food,agent_init,predator_init,initial_probability,initial_food_agents, initial_food_predators,initial_multiplicator):
    """fonction qui permettra de creer la grille en faisant appel a create_agent et set_position elle retourne la grille et le tableau des agents"""
    grid = [0]*n
    for i in range(len(grid)):
        grid[i] = [0]*n
    for i in range (len(grid)):
        for j in range(len(grid[0])):
            local_growth=max_food*random.random()
            grid[i][j]=[False,random.random()*local_growth,local_growth]  #grid[i][j]=[cell occupied,Amount of food,food_growth]
    agent_list = []
    predator_list = []

    for ag in agent_init:
        agent_list.append(ag)
        grid[ag.position[0]][ag.position[1]][0] = 1

    for pr in predator_init:
        predator_list.append(pr)
        grid[pr.position[0]][pr.position[1]][0] = 2

    for i in range(nbAgents): #Agents creation
        coord=set_position(grid)
        grid[coord[0]][coord[1]][0]=1
        agent_list.append(Agent(type=1,food=initial_food_agents,dying_threshold=10,minimum_food=100,
        position=coord,nb_offspring=1,food_transmitted=100,generation=0,
        mutation_probability=initial_probability,age=0,totalchildren=0,multiplicator=1))

    for i in range(nb_predators): #predator creation
        coord=set_position(grid)
        grid[coord[0]][coord[1]][0]=2
        predator_list.append(Agent(type=2,food=initial_food_predators,dying_threshold=10,minimum_food=170,position=coord,nb_offspring=1,food_transmitted=100,generation=0,mutation_probability=initial_probability,age=0,totalchildren=0,multiplicator=initial_multiplicator))

    return grid,agent_list,predator_list


def display(grid,agent_list,predator_list,day):
    food=[0]*len(grid)
    agents=[0]*len(grid)
    for i in range(len(grid)):
        food[i]=[0]*len(grid[0])
        agents[i]=[0]*len(grid[0])
    for i in range(len(grid)):
        for j in range (len(grid[0])):
            food[i][j]=int(grid[i][j][1])
            if grid[i][j][0]==1:
                for ag in agent_list:
                    if ag.position==[i,j]:
                        agents[i][j]="A"
            elif grid[i][j][0]==2:
                for pr in predator_list:
                    if pr.position==[i,j]:
                        agents[i][j]="P"
            else : agents[i][j]=" "
    """
    print("Food grid rounded :")
    for i in range (len(grid)):
        print(' '.join(map(str, food[i])))
    """
    print("")
    print("day :",day,"- Agent grid    :    A =",len(agent_list),"    P =",len(predator_list))
    for i in range (len(grid)):
        print(' '.join(map(str, agents[i])))
    print("")
    time.sleep(0.2)

def create_order(n):
    l = list(range(n))
    random.shuffle(l)
    return l

def evolve(grid,agent_list,predator_list,food_regeneration, alpha, beta,growth,day,period):
    """macro function making all the agents evolve, it includes moving, eating, giving birth and dying"""

    init_nb_agents=len(agent_list)
    init_nb_predators=len(predator_list)
    order_agents = create_order(init_nb_agents)
    order_predators = create_order(init_nb_predators)
    for elt in order_agents:
        agent_list[elt].change_position(grid,agent_list)
    for elt in order_predators:
        predator_list[elt].change_position(grid,agent_list)

    init_nb_agents=len(agent_list)
    init_nb_predators=len(predator_list)
    order_agents = create_order(init_nb_agents)
    order_predators = create_order(init_nb_predators)
    birth_list_agents=[0]*init_nb_agents
    birth_list_predators=[0]*init_nb_predators
    death_list_agents=[0]*init_nb_agents
    death_list_predators=[0]*init_nb_predators

    for elt in order_agents:
        evolve=agent_list[elt].check_evolution() #receiving the value corresponding to whether the agent will die or give birth
        if evolve==-1:death_list_agents[elt]=1
        if evolve>0:birth_list_agents[elt]=evolve

    for elt in order_predators:
        evolve=predator_list[elt].check_evolution() #receiving the value corresponding to whether the agent will die or give birth
        if evolve==-1:death_list_predators[elt]=1
        if evolve>0:birth_list_predators[elt]=evolve

    newborn_agents=[]
    newborn_predators=[]

    for k in range (init_nb_agents): #temporary list of new agents that will be concatenated with the list of remaining agents
        """
        if agent_list[k].food<0 :
            print("agent",k,"with negative food",agent_list[k].food,"at",agent_list[k].position,nb_neighbours(grid,agent_list[k].position))
        """
        if birth_list_agents[k]>0:
            newborn_agents+=agent_list[k].give_birth(grid,birth_list_agents[k])
    for k in range (init_nb_agents-1,-1,-1): #necessary to keep track of indexes as we pop the agents out of agent_list
        if death_list_agents[k]==1:
            grid[agent_list[k].position[0]][agent_list[k].position[1]][0]=False
            #Conservation of food
            grid[agent_list[k].position[0]][agent_list[k].position[1]][1]+=agent_list[k].food
            # if agent_list[k].food<0 :
            #     print("dropping",agent_list[k].food,"at",agent_list[k].position,nb_neighbours(grid,agent_list[k].position))
            agent_list.pop(k)
    agent_list+=newborn_agents



    for k in range (init_nb_predators): #temporary list of new agents that will be concatenated with the list of remaining agents
        """
        if predator_list[k].food<0 :
            print("predator",k,"with negative food",predator_list[k].food,"at",predator_list[k].position,nb_neighbours(grid,predator_list[k].position))
        """
        if birth_list_predators[k]>0:
            newborn_predators+=predator_list[k].give_birth(grid,birth_list_predators[k])
    for k in range (init_nb_predators-1,-1,-1): #necessary to keep track of indexes as we pop the predators out of predator_list
        if death_list_predators[k]==1:
            grid[predator_list[k].position[0]][predator_list[k].position[1]][0]=False
            #Conservation of food
            grid[predator_list[k].position[0]][predator_list[k].position[1]][1]+=predator_list[k].food
            # if predator_list[k].food<0 :
            #     print("dropping",predator_list[k].food,"at",predator_list[k].position,nb_neighbours(grid,predator_list[k].position))
            predator_list.pop(k)
    predator_list+=newborn_predators


    if growth=="LV":
        regenerate_food_LV(grid,len(agent_list),food_regeneration,alpha, beta)
    elif growth=="steady":
        regenerate_food_steady(grid,food_regeneration)
    elif growth=="seasons":
        regenerate_food_seasons(grid,food_regeneration,day,period)
    elif growth=="step" : regenerate_food_step(grid,food_regeneration,day,period)

def regenerate_food_steady(grid,food_regeneration):
    """adds on each cell food_regeneration% of the cells maxfood (ie grid[i][j][2]"""
    for i in range(len(grid)):
        for j in range (len(grid[0])):
            if grid[i][j][1]<grid[i][j][2]:grid[i][j][1]+=grid[i][j][2]*food_regeneration/100

def regenerate_food_seasons(grid,food_regeneration,day,period):
    """adds on each cell food_regeneration% of the cells maxfood (ie grid[i][j][2]"""
    for i in range(len(grid)):
        for j in range (len(grid[0])):
            if grid[i][j][1]<grid[i][j][2]:grid[i][j][1]+=grid[i][j][2]*(food_regeneration/100)*(1+0.5*np.sin(2*np.pi*day/period))

def regenerate_food_step(grid,food_regeneration,day,period):
    """adds on each cell food_regeneration% of the cells maxfood (ie grid[i][j][2]"""
    for i in range(len(grid)):
        for j in range (len(grid[0])):
            if grid[i][j][1]<grid[i][j][2] and day%(2*period)<period : grid[i][j][1] += grid[i][j][2]*(food_regeneration/100)*1.5
            elif grid[i][j][1]<grid[i][j][2] and day%(2*period)>period : grid[i][j][1] += grid[i][j][2]*(food_regeneration/100)*0.5

def regenerate_food_LV(grid,nb_agents, food_regeneration, alpha, beta):
    """
    regenerates food according to Lotka volterra equations : if the number of food is higher than the number predicted by the equation, we don't change anything.
    if it is lower, we regenerate food on random cells until right amount of food is reached
   """
    order_x = create_order(len(grid))
    order_y = create_order(len(grid[0]))
    nb_food = count_food(grid)
    new_food = alpha*nb_food - beta * nb_food*nb_agents + nb_food
    for i in order_x:
        for j in order_y:
            if grid[i][j][1]<grid[i][j][2] and nb_food <= new_food :
                grid[i][j][1]+=grid[i][j][2]*food_regeneration/100
                nb_food += grid[i][j][2]*food_regeneration/100

def count_food(grid):
    food=0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            food+=grid[i][j][1]
    return food

def print_file_nb_agents(name,nb_agents_tab): #obsolete
    """writes in a file the argument of the simulation"""
    file = open(name,"w")
    for i in range(len(nb_agents_tab)):
        file.write(str(i)+" "+str(nb_agents_tab[i]))
        file.write("\n")
    file.close()

def write_data_file(grid,day,tab_agents,tab_predators,index,parameters,finished,negativefood,sumfood,RoT):
    growth=parameters[11][1]
    if day == 0:
        os.mkdir(f"data/simulation_{index}")
        f = open(f"data/simulation_{index}/datafile.txt", "w")
        f.close()
    

    file = open(f"data/simulation_{index}/{day}.dat","w")
    if sumfood==True:
        file.write("f 0 0 "+str(count_food(grid))+" \n")
    else:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j][1] !=0 :
                    file.write("f "+ str(i)+" "+str(j)+" "+str(grid[i][j][1])+" \n")
    for agent in tab_agents:
        file.write("a "+str(agent.position[0])+" "+str(agent.position[1])+" "+str(agent.food)+" "+str(agent.dying_threshold)+" "+str(agent.minimum_food)+" "+str(agent.nb_offspring)+" "+str(agent.food_transmitted)+" "+str(agent.mutation_probability)+" "+str(agent.multiplicator)+" "+str(agent.generation)+" \n")

    for predator in tab_predators:
        file.write("p "+str(predator.position[0])+" "+str(predator.position[1])+" "+str(predator.food)+" "+str(predator.dying_threshold)+" "+str(predator.minimum_food)+" "+str(predator.nb_offspring)+" "+str(predator.food_transmitted)+" "+str(predator.mutation_probability)+" "+str(predator.multiplicator)+" "+str(predator.last_meal)+" "+str(predator.generation)+" \n")

    file.close()
    if finished==1 :
        if day!=parameters[1][1]:
            parameters[1][1]=day
        datafile = open(f"data/simulation_{index}/datafile.txt", "a")
        for i in range (len(parameters)):
            datafile.write(parameters[i][0]+" "+str(parameters[i][1])+"\n")
        datafile.write("negative_food"+" "+str(negativefood)+"\n")
        datafile.write("sumfood"+" "+str(sumfood)+"\n")
        datafile.write(RoT)
        datafile.close()

def test_constant(k,nb_1,nb):
    if nb_1==nb:k+=1
    else:k=0
    if k>1000 : return k,True
    else : return k,False

def double_check(agent_list,predator_list,grid):
    count=0
    for k in range (0,len(agent_list),-1):
        if agent_list[k].food<10:
            pos=agent_list[k].position
            agent_list.pop(k)
            grid[pos[0]][pos[1]][0]=0
            count+=1
    for k in range (0,len(predator_list),-1):
        if predator_list[k].food<10:
            pos=predator_list[k].position
            predator_list.pop(k)
            grid[pos[0]][pos[1]][0]=0
            count+=1
    return(count)
    
def run_evolution(n,duration,nbagents,nbpredators,initial_probability,max_food,food_regeneration,alpha,beta,dis,step,index,growth,period=365,init_agents=[],init_predators=[]):
    #Initiating parameters
    negativefood=0
    day=0
    max=0
    maxgen=0
    k=0
    nb_1=0
    test=False
    sumfood=True
    RoT="" #reason of termination
    totnegativefood=0
    parameters=[["n",n],["duration",duration],["nbagents",nbagents],["nbpredators",nbpredators],["initial_probability",initial_probability],["max_food",max_food],["food_regeneration",food_regeneration],["alpha",alpha],["beta",beta],["index",index],["step",step],["growth",growth],["period",period]]
    
    #Initiating the system
    grid,agent_list,predator_list = create_grid(n,nbagents,nbpredators,max_food,init_agents,init_predators,initial_probability,100,100,1.0)
    nombre_agents,nombre_predateurs,food = [],[],[]
    if nbpredators==0:nopredators=True
    else:nopredators=False

    #Looping the evolution
    while (len(agent_list)!=0 and (nopredators or len(predator_list)!=0) and max<len(grid)*len(grid[0]) and day<duration and test==False and negativefood<=1):
        
        #Checking for negative food
        count=0
        # count+=double_check(agent_list,predator_list,grid)
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j][1]<0:
                     print("negative food at ",i,j,"amount :",round(grid[i][j][1],5),"day :",day)
                     totnegativefood+=grid[i][j][1]
                     negativefood=1
        for i in range(len(agent_list)):
            if agent_list[i].food<0:negativefood=2
        for i in range(len(predator_list)):
            if predator_list[i].food<0:negativefood=3
        
        #Checking for stagnation
        nb_1=len(agent_list)
        
        #Maxgen and avgfood calculation:
        avg_food=0
        for i in range(len(agent_list)):
            avg_food+=agent_list[i].food
            if agent_list[i].generation>maxgen:maxgen=agent_list[i].generation
        
        #Displaying info
        if dis>-1 and dis<2:
            p=0
            # print('average_food :',int(avg_food/len(agent_list)))
            for i in range(len(agent_list)):
                if agent_list[i].generation==maxgen:
                    print('day:',day," | agents:",len(agent_list)," | gen:",maxgen,' | avgfood:',int(avg_food/len(agent_list))," | minfood:",round(agent_list[i].minimum_food,2),"| offspring :  ",round(agent_list[i].nb_offspring,2)," | foodtrans:",round(agent_list[i].food_transmitted,2))
                    p=1
                elif len(agent_list)<5:
                    print('day:',day," | agents:",len(agent_list)," | gen:",agent_list[i].generation,' | avgfood:',int(avg_food/len(agent_list))," | minfood:",round(agent_list[i].minimum_food,2),"| offspring :  ",round(agent_list[i].nb_offspring,2)," | foodtrans:",round(agent_list[i].food_transmitted,2))
                    p=1
            if p==0:
                print('day:',day," | agents:",len(agent_list)," | gen:",maxgen,' | avgfood:',int(avg_food/len(agent_list)))
            print('---')
        if dis>0: display(grid,agent_list,predator_list,day)
        
        #Saving data
        if index !=0 and day%step==0 :write_data_file(grid,day,agent_list,predator_list,index,parameters,finished=0,negativefood=negativefood,sumfood=sumfood,RoT=RoT) #write every step dayss
        
        #Evolving the simulation
        evolve(grid,agent_list,predator_list,food_regeneration,alpha,beta,growth,day,period)
        day+=1
        
        #Checking for negative food
        count+=double_check(agent_list,predator_list,grid)
        if count!=0:print("day :",day," - corrected :",count)
        
        #Saving more data
        if len(agent_list)>max:max=len(agent_list)
        nombre_agents.append(len(agent_list))
        nombre_predateurs.append(len(predator_list))
        food.append(count_food(grid))
        
        #Testing for stagnation
        k,test=test_constant(k,nb_1,len(agent_list))
        
    #Termination of the simulation
    
    #Report of the reason of termination
    if len(agent_list)==0 : RoT+=("Reason of termination : No agent left \n")
    if len(predator_list)==0 : RoT+=("Reason of termination : No predator left \n")
    if max>=len(grid)*len(grid[0]) : RoT+=("Reason of termination : Grid full of agent \n")
    if day>=duration : RoT+=("Reason of termination : Simulation finished \n")
    if test!=False : RoT+=("Reason of termination : Simulation stagnation \n")
    if negativefood!=0 : RoT+=("Reason of termination : Negative food"+str(negativefood)+"(1=grid/2=agent/3=predator) \n")
    print(RoT)
    
    #Final display
    if dis>0:display(grid,agent_list,predator_list,day)
    print("index:",index," | max agents:",max," | day:",day)
    print("total negative food:",totnegativefood)
    
    #Writing datafile
    parameters.append(["totnegativefood",totnegativefood])
    if index>0:write_data_file(grid,day,agent_list,predator_list,index,parameters,finished=1,negativefood=negativefood,sumfood=sumfood,RoT=RoT)
    
    return nombre_agents,day,food,nombre_predateurs,agent_list,predator_list,totnegativefood
    
    
    
#Testing:
# run_evolution(n,duration,nbagents,nbpredators,mutation_probability,max_food,food_regeneration,alpha,beta,dis,step,index,growth,period=365)

# run_evolution(50,2000,100,0,1,100,10,10,0.08,-1,10,0,"seasons",period=365)













