# import pandas lib as pd
import random
import numpy as np
import pandas as pd
import time
from statistics import mean
# from cloudfogcomputing_v3 import NoOfTask
 
print('---------------------------Differential Evolution---------------------------')
# read 2nd sheet of an excel file
path = 'Excel File/task40.xlsx'
alphaVal = 0.5

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

# def fun(x):
#     return x[0]**2 + x[1]**2

def differential_evolution(pop_size, generations, f, cr, bounds):
    # Initialize population
    pop = np.random.uniform(bounds[0], bounds[1], (pop_size, len(bounds)))

    # Iterate over generations
    for gen in range(generations):
        for i in range(pop_size):
            # Select three distinct individuals from the population
            a, b, c = random.sample(range(pop_size), 3)

            # Generate a trial vector by applying mutation and crossover
            mutant = pop[a] + 0.5 * (pop[b] - pop[c])
            trial = np.copy(pop[i])
            for j in range(len(bounds)):
                if random.random() < cr:
                    trial[j] = mutant[j]

            # Clip trial vector to bounds
            trial = np.clip(trial, bounds[0], bounds[1])

            # Evaluate fitness of trial vector
            f_trial = f(trial)
            f_old = f(pop[i])

            # Replace individual with trial vector if it has better fitness
            if f_trial < f_old:
                pop[i] = trial

    # Return the best individual and its fitness
    best_idx = np.argmin([f(p) for p in pop])
    best = pop[best_idx]
    best_fitness = f(best)
    return best, best_fitness

print(differential_evolution(100,500,fun,0.5,(11.1,50.1)))