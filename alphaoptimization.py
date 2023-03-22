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

##############################################




import numpy as np

# Define the utility function with an alpha parameter


def utility_function(x, alpha):
    return x ** alpha

def utility_function(makespan,totalcost,alpha,minTotalcost=minTotalcost,minmakespan=minmakespan):
  # alpha = 0.5
  x= (alpha*(minmakespan/makespan)) + ((1-alpha)*(minTotalcost/totalcost))
  return x

# Define the derivative of the utility function with respect to alpha


def d_utility_function(makespan,totalcost,alpha,minTotalcost=minTotalcost,minmakespan=minmakespan):
    return (minmakespan/makespan) - (minTotalcost/totalcost)

# Define the gradient descent function


def gradient_descent(makespan,totalcost,alpha, learning_rate, num_iterations,minTotalcost=minTotalcost,minmakespan=minmakespan):
    for i in range(num_iterations):
        gradient = d_utility_function(makespan,totalcost,alpha,minTotalcost=minTotalcost,minmakespan=minmakespan)
        alpha += learning_rate * gradient
        if alpha < 0:
            alpha = 0
        elif alpha > 1:
            alpha = 1
    return alpha


# Generate some random data for x
# x = np.random.rand(100)

# Initialize the value of alpha
alpha = 0.5

# Set the learning rate and number of iterations
learning_rate = 0.001
num_iterations = 1000

x = [5, 5, 5, 8, 8, 5, 5, 8, 5, 5, 8, 8, 8, 5, 5, 5, 8, 8, 8, 5, 8, 5, 8, 5, 8, 5, 5, 5, 5, 5, 5, 8, 8, 5, 5, 5, 8, 8, 5, 8]
x_ms = makeSpan(x)
x_tc = totalCost(x)

# Optimize the utility function using gradient descent
alpha_opt = gradient_descent(x_ms,x_tc, alpha, learning_rate, num_iterations)

# Evaluate the performance of the optimized utility function
utility_values = utility_function(x_ms,x_tc, alpha_opt)
average_utility = np.mean(utility_values)
print(f"Optimal alpha value: {alpha_opt}")
print(f"Average utility value: {average_utility}")
