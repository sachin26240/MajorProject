# import pandas lib as pd
import pandas as pd

# from cloudfogcomputing_v3 import NoOfTask
print('---------------------Genetic Algorithm----------------------------')
# read 2nd sheet of an excel file
alphaValue = 0.5
path = 'Excel File/task120.xlsx'
TaskDetails_DF = pd.read_excel(path, sheet_name = 'TaskDetails',index_col=0)
NodeDetails_DF = pd.read_excel(path, sheet_name = 'NodeDetails',index_col=0)
ExecutionTable_DF = pd.read_excel(path, sheet_name = 'ExecutionTable',index_col=0)
CostTable_DF = pd.read_excel(path, sheet_name = 'CostTable',index_col=0)

TaskDetails = TaskDetails_DF.values.tolist()
NodeDetails = NodeDetails_DF.values.tolist()
eTimeList = ExecutionTable_DF.values.tolist()[1:]
costList = CostTable_DF.values.tolist()[1:]

TotalNode = len(eTimeList)
NoOfTask = len(eTimeList[0])
# print("Total Nodes:",TotalNode,"Total Tasks:",NoOfTask)
print('Number of cloud nodes:',3)
print('Number of fog nodes:',10)
print('Number of tasks:',NoOfTask)

#minmakespan
lengthSum = 0
cpuRateSum = 0
for x in range(NoOfTask):
  lengthSum+=TaskDetails[x][0]

for x in range(TotalNode):
  cpuRateSum+=NodeDetails[x][0]

minmakespan = round((lengthSum*(10**3))/cpuRateSum , 4 )
print('minmakespan:',minmakespan)

#minTotalCost
costListTranspose =[[row[i] for row in costList] for i in range(len(costList[0]))]
minTotalcost = []
for i in range(NoOfTask):
  minTotalcost.append(min(costListTranspose[i]))
minTotalcost = sum(minTotalcost)
print('minTotalcost:',minTotalcost)
# minTotalcost

#utility function
def utilityFunction(makespan,totalcost,minTotalcost=minTotalcost,minmakespan=minmakespan,alpha = alphaValue):
  # alpha = 0.5
  x= (alpha*(minmakespan/makespan)) + ((1-alpha)*(minTotalcost/totalcost))
  return x


def totalCost(chrom):
  sum = 0
  for x in range(NoOfTask):
    sum+=costList[chrom[x]][x]
  return round(sum,4)

# x=[1,6,5,2,1,6,1,5,9,6]

def makeSpan(chrom):
  # NiTasks = []
  mspan = []
  for tem in range(TotalNode):
    val = 0
    tsk = [i for i,val in enumerate(chrom) if val==tem]
    # NiTasks.append([i for i,val in enumerate(x) if val==tem])
    for sac in tsk:
      val+=eTimeList[tem][sac]
    mspan.append(val)
  # print(NiTasks)
  return max(mspan)




'''------------------------------------------------------------------------------------'''

import pandas as pd
import numpy as np
import time
import copy
from statistics import mean

# ###################################################
# #-----importing Data-----------------------
# ###################################################
# Tasks_Vms_Cost=pd.read_excel("/content/DATASET.xlsx",sheet_name="COST",index_col =[0])
# Tasks_Vms_Reliability=pd.read_excel("/content/DATASET.xlsx",sheet_name="RELIABILITY",index_col =[0])

"""### The cost matrix 
The cost of task(i) on VM(j) 
"""

# Tasks_Vms_Cost

"""### The reliability matrix
The reliability of VM(j) when carry out task(i)
"""

# Tasks_Vms_Reliability

"""### Initialization settings"""

###################################################
#-----Initialization settings-----------------------
###################################################

#-----Scientific workflow & Cloud -----------------
num_task=NoOfTask           # number of Tasks
num_vm=TotalNode          # number of virtual machines
#-----Genetic Algorithm Paramaters ----------------
population_size= 100
num_iteration=500
crossover_rate= 0.9
mutation_rate= 0.01
mutation_selection_rate= 0.4
num_mutation_jobs=round(num_task*mutation_selection_rate)

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

"""### Chromosome
* Each individual or chromosome is represented as a vector of length equal to the number of tasks(1x100). 
* The values specified in this vector are in the range (1, number of virtual machines(40))
* The value corresponding to each position in the vector represents the VM to which task T is allocated. 
"""

###################################################
#-----Main-----------------------------------------
###################################################


###################################################
#-----Generate initial population------------------
###################################################
avgFitnessValue = []
avgCost = []
avgMakespan = []
for times in range(1):
    print('alpha:',alphaValue)
    # print('Dataset Time:',times+1)
    start_time = time.time()
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
            gen_r = utilityFunction(mksp,gen_c)
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
    # print('-----Results -----------------------------')
    # print("One chromosome(1x100)=",best_list[0])
    # print("[Reliability,Cost]=",best_obj[0])
    # print("------------------------------------------")

    print('The elapsed time:%s'% (time.time() - start_time))

    print('Global Best:',best_list[0])
    print('Total Cost:',totalCost(best_list[0]))
    print('Makespan:',makeSpan(best_list[0]))
    print('Optimal Function value:',utilityFunction(makeSpan(best_list[0]),totalCost(best_list[0])))
    print('----------------------------------------')
    avgFitnessValue.append(utilityFunction(makeSpan(best_list[0]),totalCost(best_list[0])))
    avgCost.append(totalCost(best_list[0]))
    avgMakespan.append(makeSpan(best_list[0]))
    # alphaValue+=0.1

# print('Average Cost:',mean(avgCost))
# print('Average Makespan:',mean(avgMakespan))
# print('Average Fitness Value:',mean(avgFitnessValue))

#----------------------------------------------End---------------------------------------------------------------------------------#
#---------------------------------------------------Gantt Chart---------------------------------------------------------#

def List2Matrix(task):
  lst2mat =[]
  for Ni in range(TotalNode):
    temp = []
    for Ti in range(NoOfTask):
      temp.append(0)
    lst2mat.append(temp)

  for Ti in range(NoOfTask):
    lst2mat[task[Ti]][Ti] = 1
  return lst2mat

# chrom = [8, 5, 6, 8, 12, 12, 1, 3, 6, 0, 0, 1, 4, 3, 7, 6, 0, 4, 7, 4, 3, 1, 9, 1, 7, 3, 3, 0, 12, 6, 11, 11, 8, 6, 1, 11, 11, 1, 2, 1, 0, 5, 12, 7, 5, 2, 7, 1, 2, 4, 1, 12, 1, 11, 9, 0, 0, 6, 12, 3, 9, 6, 3, 12, 5, 9, 0, 1, 1, 3, 0, 1, 11, 12, 0, 2, 1, 9, 7, 8, 2, 10, 0, 3, 6, 8, 10, 11, 4, 2, 6, 8, 6, 1, 6, 4, 9, 9, 4, 1, 2, 2, 4, 7, 2, 2, 3, 12, 2, 6, 9, 2, 1, 11, 6, 2, 1, 9, 9, 2]
lstmat= List2Matrix(best_list[0])
task = []
tme = []
lstmat= List2Matrix(best_list[0])
for tem in range(TotalNode):
  val = []
  tsk = [i for i,val in enumerate(best_list[0]) if val==tem]
  # NiTasks.append([i for i,val in enumerate(x) if val==tem])
  for sac in tsk:
    val.append(eTimeList[tem][sac])
    lstmat[tem][sac] = eTimeList[tem][sac]
  task.append(tsk)
  tme.append(val)
# print(task)
# print(tme)
#-----------------------------------------------------
cloudTask = []
temp = []
for i in range(1,NoOfTask+1):
  temp.append('Task_'+str(i))
cloudTask.append(temp)

nodeTask = []
temp = []
for i in range(1,TotalNode+1):
  temp.append('Node_'+str(i))
nodeTask.append(temp)

index=nodeTask
#-------------------------------------------------------------------
GanttChart_DF = pd.DataFrame(lstmat, columns =cloudTask, index = index) 
Excel_File= pd.ExcelWriter('output/GAchart.xlsx', engine = 'openpyxl')
GanttChart_DF.to_excel(Excel_File, sheet_name="GanttChart_DF", index=True)
Excel_File.close()
