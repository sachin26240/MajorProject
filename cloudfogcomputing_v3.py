# -*- coding: utf-8 -*-
"""CloudFogComputing_v3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1VabpJ1PqQ0HhniWFOn3qiKFc8A6bdSX0

## Cloud Node, Fog Node and Task Dataset Generation
***reference:*** https://www.mdpi.com/2076-3417/9/9/1730
"""

import random
import math
from statistics import mean
import time
import pandas as pd
# t1 = time.perf_counter()

Excel_File= pd.ExcelWriter("Data.xlsx", engine = 'openpyxl')

NoOfTask = 10
NoOfCloudNode = 3
NoOfFogNode = 10
TotalNode = NoOfCloudNode + NoOfFogNode

NoOf01Variable = (NoOfCloudNode+NoOfFogNode)*NoOfTask

NoOfContinuosVariable = (NoOfCloudNode+NoOfFogNode)+2

NoOfConstraint = (NoOfCloudNode+NoOfFogNode)*2+1+NoOfTask

print(NoOf01Variable,NoOfContinuosVariable,NoOfConstraint)

#cloud Node characteristic Generation
#each cloud node details
# cloudNodeDetails=[['CPU rate (MIPS)','CPU usage cost','Memory usage cost','Bandwidth usage cost']]
cloudNodeDetails=[]

for _ in range(NoOfCloudNode):
  cpuRate = random.randint(3000,5000)
  cpuCost = round(random.uniform(0.7,1.0),4)
  memCost = round(random.uniform(0.02,0.05),5)
  bandCost = round(random.uniform(0.05,0.1),4)
  cloudNode = [cpuRate,cpuCost,memCost,bandCost]
  cloudNodeDetails.append(cloudNode)

#fog Node characteristic Generation
#each fog node details
# fogNodeDetails=[['CPU rate (MIPS)','CPU usage cost','Memory usage cost','Bandwidth usage cost']]
fogNodeDetails = []
for _ in range(NoOfFogNode):
  cpuRate = random.randint(500,1500)
  cpuCost = round(random.uniform(0.1,0.4),4)
  memCost = round(random.uniform(0.01,0.03),5)
  bandCost = round(random.uniform(0.01,0.02),4)
  fogNode = [cpuRate,cpuCost,memCost,bandCost]
  fogNodeDetails.append(fogNode)

#task characteristic generation
TaskIndex = []
temp = []
for i in range(1,NoOfTask+1):
  temp.append('Task_'+str(i))
TaskIndex.append(temp)


TaskDetails = []
for _ in range(NoOfTask):
    taskInstruction=random.randint(1,100)
    # taskInstruction = 0
    taskMemory = random.randint(50,200)
    taskInputSize = random.randint(10,100)
    taskOutputsize =random.randint(10,100)
    TaskDetails.append([taskInstruction,taskMemory,taskInputSize,taskOutputsize])

# print(TaskDetails)
TaskDetails_DF=pd.DataFrame(TaskDetails,columns=['Number of instructions (109 instructions)','Memory required (MB)','Input file size (MB)','Output file size (MB)'],index=TaskIndex)
# TaskDetails_DF
TaskDetails_DF.to_excel(Excel_File, sheet_name="TaskDetails", index=True)
# Excel_File.save()
# Excel_File.close()


# merge cloud and fog
# TaskDetails = cloudTaskDetails + fogTaskDetails
NodeIndex = []
temp = []
for i in range(1,TotalNode+1):
  temp.append('Node_'+str(i))
NodeIndex.append(temp)

NodeDetails = cloudNodeDetails + fogNodeDetails

# print(NodeDetails)
NodeDetails_DF=pd.DataFrame(NodeDetails,columns=['CPU rate (MIPS)','CPU usage cost','Memory usage cost','Bandwidth usage cost'],index=NodeIndex)
# NodeDetails_DF

NodeDetails_DF.to_excel(Excel_File, sheet_name="NodeDetails", index=True)
# Excel_File.save()
# Excel_File.close()

# executionTime Dataframe
eTimeList= []
for Ni in range(TotalNode):
  temp7 = []
  for Tk in range(NoOfTask):
    x = round((TaskDetails[Tk][0] * (10**3)) / (NodeDetails[Ni][0]),4)
    temp7.append(x)
 
  eTimeList.append(temp7)

# eTimeList
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
ExecutionTable_DF = pd.DataFrame(eTimeList, columns =cloudTask, index = index) 
# ExecutionTable_DF
ExecutionTable_DF.to_excel(Excel_File, sheet_name="ExecutionTable", index=True)
# Excel_File.save()
# Excel_File.close()

# Cost Dataframe
costList = []
for Ni in range(TotalNode):
  temp4 = []
  for Tk in range(NoOfTask):
    cpuCost = NodeDetails[Ni][1]*eTimeList[Ni][Tk]
    memoryCost = (NodeDetails[Ni][2]*TaskDetails[Tk][1])
    bandwidthCost = NodeDetails[Ni][3]*(TaskDetails[Tk][2]+TaskDetails[Tk][3])
    # temp5 = (cpuCost + memoryCost + bandwidthCost) * TaskDetails[Ni][Tk][0]
    temp6 = cpuCost + memoryCost + bandwidthCost
    # print(temp5)
    # temp4+=temp5
    temp4.append(round(temp6,4))
  costList.append(temp4)
# costList
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
CostTable_DF = pd.DataFrame(costList, columns =cloudTask, index = index) 
# CostTable_DF
CostTable_DF.to_excel(Excel_File, sheet_name="CostTable", index=True)
# Excel_File.save()
Excel_File.close()

#minmakespan
lengthSum = 0
cpuRateSum = 0
for x in range(NoOfTask):
  lengthSum+=TaskDetails[x][0]

for x in range(TotalNode):
  cpuRateSum+=NodeDetails[x][0]

minmakespan = round((lengthSum*(10**3))/cpuRateSum , 4 )
# minmakespan

#minTotalCost
costListTranspose =[[row[i] for row in costList] for i in range(len(costList[0]))]
minTotalcost = []
for i in range(NoOfTask):
  minTotalcost.append(min(costListTranspose[i]))
minTotalcost = sum(minTotalcost)
# minTotalcost

#utility function
def utilityFunction(makespan,totalcost,minTotalcost=minTotalcost,minmakespan=minmakespan,alpha = 0.5):
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




"""## Optimizing the Cost for Cloud-Fog System Using NSGA-II




"""

# !pip install openpyxl==3.0.9

###################################################
#-----importing library---------------------------
###################################################
import pandas as pd
import numpy as np
import time
import copy

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
num_task=10           # number of Tasks
num_vm=13             # number of virtual machines
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
start_time = time.time()

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
print('-----Results -----------------------------')
print("One chromosome(1x100)=",best_list[0])
print("[Reliability,Cost]=",best_obj[0])
print("------------------------------------------")

print('The elapsed time:%s'% (time.time() - start_time))

print('Global Best:',best_list[0])
print('Total Cost:',totalCost(best_list[0]))
print('Makespan:',makeSpan(best_list[0]))
print('Optimal Function value:',utilityFunction(makeSpan(best_list[0]),totalCost(best_list[0])))
print('----------------------------------------')

"""### Pareto optimal of Reliability and Cost
We got Pareto optimal front as shown in the next figure (Red points) after executing 500 generations.

**Note:** We generated 2000 random solutions (Blue points) and added them to the chart to compare with the final generation.
"""

###################################################
#-----Generate 1000 random chromosome --------------
###################################################
data_list=[]
for i in range(2000):
    nxm_random_num=list(np.random.permutation(num_task))
    data_list.append(nxm_random_num) 
    for j in range(num_task):
        data_list[i][j]=data_list[i][j]%num_vm 
        
    '''--------fitness value(calculate  makespan and TWET)-------------'''
data_obj_record={} 
for m in range(1000):
    gen_c=0
    gen_r=1
    #for i in total_chromosome[m]:
    # for nn in range(num_task):
    #     gen_c +=Task_Vm_Cost[nn][data_list[m][nn]]
    #     gen_r *=Task_Vm_Reliability[nn][data_list[m][nn]]
    mksp = makeSpan(data_list[m])
    gen_c = totalCost(data_list[m])
    gen_r = utilityFunction(mksp,gen_c)
    data_obj_record[m]=[gen_r,gen_c]  

data_obj_list=[data_obj_record[k] for k in data_obj_record]

###################################################
#-----Pareto optimal of Reliability and Cost ------
###################################################

import matplotlib.pyplot as plt
a = np.array(best_obj)
b = np.array(data_obj_list)
plt.figure(figsize=(8,6))
plt.xlabel('Reliability', fontsize=15)
plt.ylabel('Cost', fontsize=15)
plt.title("Pareto optimal of Reliability and Cost.")
plt.scatter(b[:,0], b[:,1], s=15)
plt.scatter(a[:,0], a[:,1], c='red', s=15)

"""### References:
1. Jiang, Zengqiang, and Zuo Le. "Study on multi-objective flexible job-shop scheduling problem considering energy consumption." Journal of Industrial Engineering and Management (JIEM) 7, no. 3 (2014): 589-604.
2. Subashini, G., and M. C. Bhuvaneswari. "Comparison of multi-objective evolutionary approaches for task scheduling in distributed computing systems." Sadhana 37, no. 6 (2012): 675-694.
3. Nidhiry, N. M., and R. Saravanan. "Scheduling optimization of a flexible manufacturing system using a modified NSGA-II algorithm." Advances in Production Engineering & Management 9, no. 3 (2014): 139-151.
4. https://github.com/wurmen/Genetic-Algorithm-for-Job-Shop-Scheduling-and-NSGA-II
"""



"""# modified particle swarm optimization"""

# task = [1,0,7,9,5,4,3,8,6,1]
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

# sach=List2Matrix(task)

# sachin = sach
def Matrix2List(task):
  mat2lst = []
  for Ti in range(NoOfTask):
    for Ni in range(TotalNode):
      if task[Ni][Ti] == 1:
        mat2lst.append(Ni)
  return mat2lst

# Matrix2List(sachin)

def velocityMatix(rang):
  mat =[]
  for Ni in range(TotalNode):
    temp = []
    for Ti in range(NoOfTask):
      temp.append(round(random.uniform(-rang,rang),8))
    mat.append(temp)
  return mat

# velocityMatix(Vmax)

start_time = time.time()

num_task = 10
num_population = 100
num_iteration = 500
c1 = 1.5
c2 = 1.5
w= 0.6
Vmax = 28

population_list=[]
for i in range(num_population):
    #print('i=',i)
    nxm_random_num=list(np.random.permutation(num_task)) # generate a random permutation of 0 to num_job*num_mc-1
    population_list.append(nxm_random_num) # add to the population_list
    for j in range(num_task):
        population_list[i][j]=population_list[i][j]%num_vm # convert to job number format, every job appears m times
# population_list

# lst=[4,8,6,3,7,9,2,1,2,9]
vel = []
lst2mat = []
# vel = velocityMatix(Vmax)
# lst2mat = List2Matrix(lst)
for i in range(num_population):
  vel.append(velocityMatix(Vmax))
  lst2mat.append(List2Matrix(population_list[i]))

pbest = population_list
gbest = population_list[random.randint(0,num_population-1)]
#loop
for i in range(num_iteration):
  # print('iteration:',i)
  for particle in range(num_population):
    mat2lst = Matrix2List(lst2mat[particle])
    # print(mat2lst)
    tc = totalCost(mat2lst)
    ms= makeSpan(mat2lst)
    fx = utilityFunction(ms,tc)
    pb = utilityFunction(makeSpan(pbest[particle]),totalCost(pbest[particle]))
    if fx > pb:
      pbest[particle] = mat2lst
      pb=fx

    gb = utilityFunction(makeSpan(gbest),totalCost(gbest))
    # print('before gbest',pb,gb)
    if pb > gb:
      gbest = pbest[particle]
      gb = pb

  #update velocity
  for particle in range(num_population):
    pbest2Mat = List2Matrix(pbest[particle])
    gbest2Mat = List2Matrix(gbest)
    for Ni in range(TotalNode):
      # temp = []
      for Ti in range(NoOfTask):
        vel[particle][Ni][Ti] = (w*vel[particle][Ni][Ti])+((c1*0.4)*(pbest2Mat[Ni][Ti]-lst2mat[particle][Ni][Ti]))+((c2*0.6)*(gbest2Mat[Ni][Ti]-lst2mat[particle][Ni][Ti]))
    
    #update position
    velTranspose =[[row[i] for row in vel[particle]] for i in range(len(vel[particle][0]))]
    for Ti in range(NoOfTask):
      for Ni in range(TotalNode):
        if Ni == velTranspose[Ti].index(max(velTranspose[Ti])):
          lst2mat[particle][Ni][Ti] = 1
        else:
          lst2mat[particle][Ni][Ti] = 0
    # print(lst2mat)
print('Global best:',gbest)
print('Total Cost:',totalCost(gbest))
print('Makespan:',makeSpan(gbest))
print('Optimal Function value:',utilityFunction(makeSpan(gbest),totalCost(gbest)))
print('The elapsed time:%s'% (time.time() - start_time))

# utilityFunction(makeSpan([4, 6, 1, 9, 4, 8, 7, 0, 3, 0]),totalCost([4, 6, 1, 9, 4, 8, 7, 0, 3, 0]))

# print(makeSpan([4, 6, 1, 9, 4, 8, 7, 0, 3, 0]))
# # totalCost([4, 6, 1, 9, 4, 8, 7, 0, 3, 0])