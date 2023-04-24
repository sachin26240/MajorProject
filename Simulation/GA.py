import pandas as pd
import numpy as np
import time
import copy
from statistics import mean


# speed up the data search
# Task_Vm_Cost=[list(map(float, Tasks_Vms_Cost.iloc[i])) for i in range(num_task)]
# Task_Vm_Reliability=[list(map(float,Tasks_Vms_Reliability.iloc[i])) for i in range(num_task)]

###################################################
#-----Non-dominated sorting function---------------
###################################################

def non_dominated_sorting(population_size,chroms_obj_record):
    s,n={},{}
    front,rank={},{}
    front[0]=[]     
    for p in range(population_size*2):
        s[p]=[]
        n[p]=0
        for q in range(population_size*2):
            
            if ((chroms_obj_record[p][0]>chroms_obj_record[q][0] and chroms_obj_record[p][1]<chroms_obj_record[q][1]) 
            or (chroms_obj_record[p][0]>=chroms_obj_record[q][0] and chroms_obj_record[p][1]<chroms_obj_record[q][1])
            or (chroms_obj_record[p][0]>chroms_obj_record[q][0] and chroms_obj_record[p][1]<=chroms_obj_record[q][1])):
                if q not in s[p]:
                    s[p].append(q)
            elif ((chroms_obj_record[p][0]<chroms_obj_record[q][0] and chroms_obj_record[p][1]>chroms_obj_record[q][1]) 
            or (chroms_obj_record[p][0]<=chroms_obj_record[q][0] and chroms_obj_record[p][1]>chroms_obj_record[q][1])
            or (chroms_obj_record[p][0]<chroms_obj_record[q][0] and chroms_obj_record[p][1]>=chroms_obj_record[q][1])):
                n[p]=n[p]+1
        if n[p]==0:
            rank[p]=0
            if p not in front[0]:
                front[0].append(p)
    
    i=0
    while (front[i]!=[]):
        Q=[]
        for p in front[i]:
            for q in s[p]:
                n[q]=n[q]-1
                if n[q]==0:
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i=i+1
        front[i]=Q
                
    del front[len(front)-1]
    return front

###################################################
#-----Calculate crowding distance function---------
###################################################

def calculate_crowding_distance(front,chroms_obj_record):
    
    distance={m:0 for m in front}
    for o in range(2):
        obj={m:chroms_obj_record[m][o] for m in front}
        sorted_keys=sorted(obj, key=obj.get)
        distance[sorted_keys[0]]=distance[sorted_keys[len(front)-1]]=999999999999
        for i in range(1,len(front)-1):
            if len(set(obj.values()))==1:
                distance[sorted_keys[i]]=distance[sorted_keys[i]]
            else:
                distance[sorted_keys[i]]=distance[sorted_keys[i]]+(obj[sorted_keys[i+1]]-obj[sorted_keys[i-1]])/(obj[sorted_keys[len(front)-1]]-obj[sorted_keys[0]])
            
    return distance  

###################################################
#-----Selection------------------------------------
###################################################
def selection(population_size,front,chroms_obj_record,total_chromosome):   
    N=0
    new_pop=[]
    while N < population_size:
        for i in range(len(front)):
            N=N+len(front[i])
            if N > population_size:
                distance=calculate_crowding_distance(front[i],chroms_obj_record)
                sorted_cdf=sorted(distance, key=distance.get)
                sorted_cdf.reverse()
                for j in sorted_cdf:
                    if len(new_pop)==population_size:
                        break                
                    new_pop.append(j)              
                break
            else:
                new_pop.extend(front[i])
    
    population_list=[]
    for n in new_pop:
        population_list.append(total_chromosome[n])
    
    return population_list,new_pop


def Genetic_Algorithm(pop_size, generations, f, NoOfTask,TotalNode, makeSpan, totalCost):
    ###################################################
    #-----Main-----------------------------------------
    ###################################################
    #-----Scientific workflow & Cloud -----------------
    num_task=NoOfTask           # number of Tasks
    num_vm=TotalNode          # number of virtual machines
    #-----Genetic Algorithm Paramaters ----------------
    population_size= pop_size
    num_iteration=generations
    crossover_rate= 0.9
    mutation_rate= 0.01
    mutation_selection_rate= 0.4
    num_mutation_jobs=round(num_task*mutation_selection_rate)

    ###################################################
    #-----Generate initial population------------------
    ###################################################
    best_list,best_obj=[],[]
    population_list=[]
    for i in range(population_size):
        #print('i=',i)
        nxm_random_num=list(np.random.permutation(num_task)) # generate a random permutation of 0 to num_job*num_mc-1
        population_list.append(nxm_random_num) # add to the population_list
        for j in range(num_task):
            population_list[i][j]=population_list[i][j]%num_vm # convert to job number format, every job appears m times

            
    for n in range(num_iteration):         
        ###################################################
        #-----Crossover------------------------------------
        ###################################################
        parent_list=copy.deepcopy(population_list)
        offspring_list=[]
        S=list(np.random.permutation(population_size)) # generate a random sequence to select the parent chromosome to crossover
        
        for m in range(int(population_size/2)):
            
            parent_1= population_list[S[2*m]][:]
            parent_2= population_list[S[2*m+1]][:]
            child_1=parent_1[:]
            child_2=parent_2[:]
            
            cutpoint=list(np.random.choice(num_task, 2, replace=False))
            cutpoint.sort()
        
            child_1[cutpoint[0]:cutpoint[1]]=parent_2[cutpoint[0]:cutpoint[1]]
            child_2[cutpoint[0]:cutpoint[1]]=parent_1[cutpoint[0]:cutpoint[1]]
            
            offspring_list.extend((child_1,child_2))

        ###################################################
        #-----Mutation-------------------------------------
        ###################################################
        for m in range(len(offspring_list)):
            mutation_prob=np.random.rand()
            if mutation_rate <= mutation_prob:
                m_chg=list(np.random.choice(num_task, num_mutation_jobs, replace=False)) # chooses the position to mutation
                t_value_last=offspring_list[m][m_chg[0]] # save the value which is on the first mutation position
                for i in range(num_mutation_jobs-1):
                    offspring_list[m][m_chg[i]]=offspring_list[m][m_chg[i+1]] # displacement
                
                offspring_list[m][m_chg[num_mutation_jobs-1]]=t_value_last
                    
        ###################################################
        #-----Fitness valuse ------------------------------
        ###################################################               
        total_chromosome=copy.deepcopy(parent_list)+copy.deepcopy(offspring_list)
        chroms_obj_record={} 
        for m in range(population_size*2):
            gen_c=0
            gen_r=1
            mksp = makeSpan(total_chromosome[m])
            gen_c = totalCost(total_chromosome[m])
            gen_r = f(mksp,gen_c)
            # for nn in range(num_task):

            #     gen_c +=Task_Vm_Cost[nn][total_chromosome[m][nn]]
            #     gen_r *=Task_Vm_Reliability[nn][total_chromosome[m][nn]]
            chroms_obj_record[m]=[gen_r,gen_c]
            
        ###################################################
        #-----Non-dominated sorting -----------------------
        ################################################### 
        front=non_dominated_sorting(population_size,chroms_obj_record)
        
        ###################################################
        #-----Selection -----------------------------------
        ###################################################         
        population_list,new_pop=selection(population_size,front,chroms_obj_record,total_chromosome)
        new_pop_obj=[chroms_obj_record[k] for k in new_pop] 
        
        ###################################################
        #-----Comparison ----------------------------------
        ################################################### 
        if n==0:
            best_list=copy.deepcopy(population_list)
            best_obj=copy.deepcopy(new_pop_obj)
        else:            
            total_list=copy.deepcopy(population_list)+copy.deepcopy(best_list)
            total_obj=copy.deepcopy(new_pop_obj)+copy.deepcopy(best_obj)
            
            now_best_front=non_dominated_sorting(population_size,total_obj)
            best_list,best_pop=selection(population_size,now_best_front,total_obj,total_list)
            best_obj=[total_obj[k] for k in best_pop]
    ###################################################
    #-----Results ------------------------------------
    ###################################################

    print('Global Best:',best_list[0])
    print('Total Cost:',totalCost(best_list[0]))
    print('Makespan:',makeSpan(best_list[0]))
    print('Optimal Function value:',f(makeSpan(best_list[0]),totalCost(best_list[0])))
    print('----------------------------------------')
    return best_list[0] , f(makeSpan(best_list[0]),totalCost(best_list[0]))

