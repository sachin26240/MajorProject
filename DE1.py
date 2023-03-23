# import pandas lib as pd
import random
import numpy as np
import pandas as pd
import time
from statistics import mean
# from cloudfogcomputing_v3 import NoOfTask
 
print('---------------------------Differential Evolution---------------------------')
# read 2nd sheet of an excel file
path = 'Excel File/task120.xlsx'
alphaVal = 1.0

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
def utilityFunction(makespan,totalcost,minTotalcost=minTotalcost,minmakespan=minmakespan,alpha = alphaVal):
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


#############################################
import random
import numpy as np

def fun(x):
    return x[0]**2 + x[1]**2

def differential_evolution(pop_size, generations, f, cr, bounds):
    # Initialize population
    pop = np.random.randint(bounds[0], bounds[1], (pop_size, NoOfTask))
    # print(bounds[0], bounds[1])
    # print(bounds)
    # print(pop)
    # Iterate over generations
    for _ in range(generations):
        for i in range(pop_size):
            # Select three distinct individuals from the population
            a, b, c = random.sample(range(pop_size), 3)
            # print('a,b,c:',a,b,c)
            # Generate a trial vector by applying mutation and crossover
            mutant = pop[a] + 0.5 * (pop[b] - pop[c])
            # print('mutation:',mutant)
            trial = np.copy(pop[i])
            
            for j in range(NoOfTask):
                temp = random.uniform(0,1)
                if temp < cr:
                    trial[j] = mutant[j]
                    # print('in',temp,j)
            # print('trail:',trial)

            # Clip trial vector to bounds
            trial = np.clip(trial, bounds[0], bounds[1])
            # print('trail:',trial)

            # Evaluate fitness of trial vector
            f_trial = f(makeSpan(trial),totalCost(trial))
            f_old = f(makeSpan(pop[i]),totalCost(pop[i]))

            # Replace individual with trial vector if it has better fitness
            if f_trial > f_old:
                pop[i] = trial

    # Return the best individual and its fitness
    best_idx = np.argmin([f(makeSpan(p),totalCost(p)) for p in pop])
    best = pop[best_idx]
    best_fitness = f(makeSpan(best),totalCost(best))
    return best, best_fitness

start_time = time.time()
avgFitnessValue = []
avgCost = []
avgMakespan = []
for i in range(1):
  print(i)
  best, best_fun = differential_evolution(100,500,utilityFunction,0.5,(0,TotalNode-1))
  avgFitnessValue.append(best_fun)
  avgCost.append(totalCost(best))
  avgMakespan.append(makeSpan(best))
print('The elapsed time:%s'% (time.time() - start_time))
print('Best:',list(best))
print('Total Cost:',totalCost(best))
print('Makespan:',makeSpan(best))
print('Optimal Function value:',utilityFunction(makeSpan(best),totalCost(best)))
print('alpha:',alphaVal)
print(best_fun)
print('Average Cost:',mean(avgCost))
print('Average Fitness Value:',mean(avgFitnessValue))
print('Average MakeSpan:',mean(avgMakespan))


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
lstmat= List2Matrix(best)
task = []
tme = []
lstmat= List2Matrix(best)
for tem in range(TotalNode):
  val = []
  tsk = [i for i,val in enumerate(best) if val==tem]
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
Excel_File= pd.ExcelWriter('output/DEchart.xlsx', engine = 'openpyxl')
GanttChart_DF.to_excel(Excel_File, sheet_name="DE_GanttChart", index=True)
Excel_File.close()
