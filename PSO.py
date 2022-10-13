# import pandas lib as pd
import random
import numpy as np
import pandas as pd
import time
from statistics import mean
# from cloudfogcomputing_v3 import NoOfTask
 
print('---------------------------PSO---------------------------')
# read 2nd sheet of an excel file
path = 'Excel File/task80.xlsx'

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
# print(TotalNode,NoOfTask)

#minmakespan
lengthSum = 0
cpuRateSum = 0
for x in range(NoOfTask):
  lengthSum+=TaskDetails[x][0]

for x in range(TotalNode):
  cpuRateSum+=NodeDetails[x][0]

minmakespan = round((lengthSum*(10**3))/cpuRateSum , 4 )
print('minmakespan:',minmakespan)

# minmakespan

#minTotalCost
costListTranspose =[[row[i] for row in costList] for i in range(len(costList[0]))]
minTotalcost = []
for i in range(NoOfTask):
  minTotalcost.append(min(costListTranspose[i]))
minTotalcost = sum(minTotalcost)
print('minTotalcost:',minTotalcost)
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


#---------------------------------------------------------------------------------------------

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
avgFitnessValue = []
avgCost = []
avgMakespan = []
for times in range(20):
  print('Dataset Time:',times+1)
  start_time = time.time()

  num_task = NoOfTask
  num_vm = TotalNode
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
  avgFitnessValue.append(utilityFunction(makeSpan(gbest),totalCost(gbest)))
  avgCost.append(totalCost(gbest))
  avgMakespan.append(makeSpan(gbest))

print('Average Cost:',mean(avgCost))
print('Average Makespan:',mean(avgMakespan))
print('Average Fitness Value:',mean(avgFitnessValue))


# utilityFunction(makeSpan([4, 6, 1, 9, 4, 8, 7, 0, 3, 0]),totalCost([4, 6, 1, 9, 4, 8, 7, 0, 3, 0]))

# print(makeSpan([4, 6, 1, 9, 4, 8, 7, 0, 3, 0]))
# # totalCost([4, 6, 1, 9, 4, 8, 7, 0, 3, 0])