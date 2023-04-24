# datageneration take input as no of tasks
# display each and every nodes detail and task details
# give choice to select algorithm and run
# give which task allocated to which node and give diffenernt color
# calculate the cost and time

# future scope try to give options to add custom user tasks


import random
import math
from statistics import mean
import time
import pandas as pd
import numpy as np
# t1 = time.perf_counter()

#########################
from DE import differential_evolution as DE
from GA import Genetic_Algorithm as GA
from MPSO import PSO
#########################


class Simulation:
    def __init__(self, n, c, f) -> None:
        self.TaskDetails = []
        self.NodeDetails = []
        self.eTimeList = []
        self.costList = []
        self.minTotalcost = 0
        self.minmakespan = 0
        self.alphaVal = 0.5
        self.dataGeneration(n, c, f)
        self.TotalNode = len(self.eTimeList)
        self.NoOfTask = len(self.eTimeList[0])
        print()
        self.showNodeDetails()
        print()
        self.showTaskDetails()
        print('***********************************************************\nOptimal Values')
        self.calculateMinMakeSpan()
        self.calculateMinTotalCost()
        print()

        display = True

        while True:
            print('*******************************************************')
            start_time = time.time()

            print('Choose an Algorithm:\n1. Genetic Algorithm\n2. Particle Swarm Optimization Algorithm\n3. Differential Evolution Algorithm\n4. Exit')
            selAlgo = int(input('Choose an algorithm (1/2/3/4):'))

            # best, best_fun = self.differential_evolution(100,500,self.utilityFunction,0.5,(0,self.TotalNode-1))
            match selAlgo:
                case 1:
                    print()
                    print('******************Genetic Algorithm****************************')
                    best, best_fun = GA(100, 500, self.utilityFunction,self.NoOfTask,self.TotalNode,self.makeSpan,self.totalCost)
                case 2:
                    print()
                    print('******************Particle Swarm Optimization Algorithm****************************')
                    best, best_fun = PSO(100, 500, self.utilityFunction, self.NoOfTask,self.TotalNode,self.makeSpan,self.totalCost)
                case 3:
                    print()
                    print('******************Differential Evolution Algorithm****************************')
                    best, best_fun = DE(100, 500, self.utilityFunction, 0.5, (0, self.TotalNode-1),self.NoOfTask,self.makeSpan,self.totalCost)
                case 4:
                    print('Simulation Ended')
                    break
                case _:
                    display=False
                    print('******************************************************')
                    print('Incorrect Option')

            if display:
                print('The elapsed time:%s' % (time.time() - start_time))
                print('Alpha value:', self.alphaVal)
                print('Global Best:', list(best))
                print('Optimal Function value:',best_fun)
                print('Total Cost required:', self.totalCost(best))
                print('Makespan (Total time required):', self.makeSpan(best))
                
                print()
                self.TaskAllocation(best)
                print()
                self.GroupbyNodeAllocation(best)
                print()

    # datageneration function
    def dataGeneration(self, numoftask, numofcloudnode, numoffognode):
        # Excel_File= pd.ExcelWriter(path, engine = 'openpyxl')

        NoOfTask = numoftask
        NoOfCloudNode = numofcloudnode
        NoOfFogNode = numoffognode
        TotalNode = NoOfCloudNode + NoOfFogNode

        NoOf01Variable = (NoOfCloudNode+NoOfFogNode)*NoOfTask

        NoOfContinuosVariable = (NoOfCloudNode+NoOfFogNode)+2

        NoOfConstraint = (NoOfCloudNode+NoOfFogNode)*2+1+NoOfTask

        # print(NoOf01Variable,NoOfContinuosVariable,NoOfConstraint)

        # cloud Node characteristic Generation
        # each cloud node details
        # cloudNodeDetails=[['CPU rate (MIPS)','CPU usage cost','Memory usage cost','Bandwidth usage cost']]
        cloudNodeDetails = []

        for _ in range(NoOfCloudNode):
            cpuRate = random.randint(3000, 5000)
            cpuCost = round(random.uniform(0.7, 1.0), 4)
            memCost = round(random.uniform(0.02, 0.05), 5)
            bandCost = round(random.uniform(0.05, 0.1), 4)
            cloudNode = [cpuRate, cpuCost, memCost, bandCost]
            cloudNodeDetails.append(cloudNode)

        # fog Node characteristic Generation
        # each fog node details
        # fogNodeDetails=[['CPU rate (MIPS)','CPU usage cost','Memory usage cost','Bandwidth usage cost']]
        fogNodeDetails = []
        for _ in range(NoOfFogNode):
            cpuRate = random.randint(500, 1500)
            cpuCost = round(random.uniform(0.1, 0.4), 4)
            memCost = round(random.uniform(0.01, 0.03), 5)
            bandCost = round(random.uniform(0.01, 0.02), 4)
            fogNode = [cpuRate, cpuCost, memCost, bandCost]
            fogNodeDetails.append(fogNode)

        # task characteristic generation
        TaskIndex = []
        temp = []
        for i in range(1, NoOfTask+1):
            temp.append('Task_'+str(i))
        TaskIndex.append(temp)

        # TaskDetails = []
        for _ in range(NoOfTask):
            taskInstruction = random.randint(1, 100)
            # taskInstruction = 0
            taskMemory = random.randint(50, 200)
            taskInputSize = random.randint(10, 100)
            taskOutputsize = random.randint(10, 100)
            self.TaskDetails.append(
                [taskInstruction, taskMemory, taskInputSize, taskOutputsize])

        # print(TaskDetails)
        # TaskDetails_DF=pd.DataFrame(TaskDetails,columns=['Number of instructions (109 instructions)','Memory required (MB)','Input file size (MB)','Output file size (MB)'],index=TaskIndex)
        # TaskDetails_DF
        # TaskDetails_DF.to_excel(Excel_File, sheet_name="TaskDetails", index=True)
        # Excel_File.save()
        # Excel_File.close()

        # merge cloud and fog
        # TaskDetails = cloudTaskDetails + fogTaskDetails
        NodeIndex = []
        temp = []
        for i in range(1, TotalNode+1):
            temp.append('Node_'+str(i))
        NodeIndex.append(temp)

        self.NodeDetails = cloudNodeDetails + fogNodeDetails

        # print(NodeDetails)
        # NodeDetails_DF=pd.DataFrame(NodeDetails,columns=['CPU rate (MIPS)','CPU usage cost','Memory usage cost','Bandwidth usage cost'],index=NodeIndex)
        # NodeDetails_DF

        # NodeDetails_DF.to_excel(Excel_File, sheet_name="NodeDetails", index=True)
        # Excel_File.save()
        # Excel_File.close()

        # executionTime Dataframe
        # eTimeList= []
        for Ni in range(TotalNode):
            temp7 = []
            for Tk in range(NoOfTask):
                x = round(
                    (self.TaskDetails[Tk][0] * (10**3)) / (self.NodeDetails[Ni][0]), 4)
                temp7.append(x)

            self.eTimeList.append(temp7)

        # eTimeList
        cloudTask = []
        temp = []
        for i in range(1, NoOfTask+1):
            temp.append('Task_'+str(i))
        cloudTask.append(temp)

        nodeTask = []
        temp = []
        for i in range(1, TotalNode+1):
            temp.append('Node_'+str(i))
        nodeTask.append(temp)

        # index=nodeTask
        # ExecutionTable_DF = pd.DataFrame(self.eTimeList, columns =cloudTask, index = index)
        # print(ExecutionTable_DF)
        # ExecutionTable_DF
        # ExecutionTable_DF.to_excel(Excel_File, sheet_name="ExecutionTable", index=True)
        # Excel_File.save()
        # Excel_File.close()

        # Cost Dataframe
        # costList = []
        for Ni in range(TotalNode):
            temp4 = []
            for Tk in range(NoOfTask):
                cpuCost = self.NodeDetails[Ni][1]*self.eTimeList[Ni][Tk]
                memoryCost = (self.NodeDetails[Ni][2]*self.TaskDetails[Tk][1])
                bandwidthCost = self.NodeDetails[Ni][3] * \
                    (self.TaskDetails[Tk][2]+self.TaskDetails[Tk][3])
                # temp5 = (cpuCost + memoryCost + bandwidthCost) * TaskDetails[Ni][Tk][0]
                temp6 = cpuCost + memoryCost + bandwidthCost
                # print(temp5)
                # temp4+=temp5
                temp4.append(round(temp6, 4))
            self.costList.append(temp4)
        # costList
        cloudTask = []
        temp = []
        for i in range(1, NoOfTask+1):
            temp.append('Task_'+str(i))
        cloudTask.append(temp)

        nodeTask = []
        temp = []
        for i in range(1, TotalNode+1):
            temp.append('Node_'+str(i))
        nodeTask.append(temp)

        # index=nodeTask
        # CostTable_DF = pd.DataFrame(costList, columns =cloudTask, index = index)
        # CostTable_DF
        # CostTable_DF.to_excel(Excel_File, sheet_name="CostTable", index=True)
        # Excel_File.save()
        # Excel_File.close()

    # Function to show node details
    def showNodeDetails(self):
        print('*********************Node Details************************')
        for i in range(len(self.NodeDetails)):
            temp = self.NodeDetails[i]
            print("Node_"+str(i+1))
            print('CPU rate (MIPS):', temp[0])
            print('CPU usage cost:', temp[1])
            print('Memory usage cost:', temp[2])
            print('Bandwidth usage cost:', temp[3], '\n')

    # Function to show task details
    def showTaskDetails(self):
        print('*********************Task Details************************')
        for i in range(len(self.TaskDetails)):
            temp = self.TaskDetails[i]
            print("Task_"+str(i+1))
            print('Number of instructions (10^9 instructions):', temp[0])
            print('Memory required (MB):', temp[1])
            print('Input file size (MB):', temp[2])
            print('Output file size (MB):', temp[3], '\n')

    # Function to Calculate MinMakeSpan
    def calculateMinMakeSpan(self):
        # print(TotalNode,NoOfTask)
        # print('Number of cloud nodes:',3)
        # print('Number of fog nodes:',10)
        # print('Number of tasks:',NoOfTask)

        # minmakespan
        lengthSum = 0
        cpuRateSum = 0
        for x in range(self.NoOfTask):
            lengthSum += self.TaskDetails[x][0]

        for x in range(self.TotalNode):
            cpuRateSum += self.NodeDetails[x][0]

        self.minmakespan = round((lengthSum*(10**3))/cpuRateSum, 4)
        print('minmakespan:', self.minmakespan)

        # minmakespan

    # Function to Calculate MinTotalCost
    def calculateMinTotalCost(self):
        # minTotalCost
        costListTranspose = [[row[i] for row in self.costList]
                             for i in range(len(self.costList[0]))]
        minTotalcost = []
        for i in range(self.NoOfTask):
            minTotalcost.append(min(costListTranspose[i]))
        self.minTotalcost = sum(minTotalcost)
        print('minTotalcost:', self.minTotalcost)
        # minTotalcost

    # utility function
    def utilityFunction(self, makespan, totalcost):
        # alpha = 0.5
        alpha = self.alphaVal
        x = (alpha*(self.minmakespan/makespan)) + \
            ((1-alpha)*(self.minTotalcost/totalcost))
        return x

    # TotalCost Function
    def totalCost(self, chrom):
        sum = 0
        for x in range(self.NoOfTask):
            sum += self.costList[chrom[x]][x]
        return round(sum, 4)

    # MakeSpan Function
    def makeSpan(self, chrom):
        # NiTasks = []
        mspan = []
        for tem in range(self.TotalNode):
            val = 0
            tsk = [i for i, val in enumerate(chrom) if val == tem]
            # NiTasks.append([i for i,val in enumerate(x) if val==tem])
            for sac in tsk:
                val += self.eTimeList[tem][sac]
                mspan.append(val)
        # print(NiTasks)
        return max(mspan)

    def fun(self, x):
        return x[0]**2 + x[1]**2

    # Optimization Algorithm
    def differential_evolution(self, pop_size, generations, f, cr, bounds):
        # Initialize population
        pop = np.random.randint(
            bounds[0], bounds[1], (pop_size, self.NoOfTask))
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

                for j in range(self.NoOfTask):
                    temp = random.uniform(0, 1)
                    if temp < cr:
                        trial[j] = mutant[j]
                        # print('in',temp,j)
                # print('trail:',trial)

                # Clip trial vector to bounds
                trial = np.clip(trial, bounds[0], bounds[1])
                # print('trail:',trial)

                # Evaluate fitness of trial vector
                f_trial = f(self.makeSpan(trial), self.totalCost(trial))
                f_old = f(self.makeSpan(pop[i]), self.totalCost(pop[i]))

                # Replace individual with trial vector if it has better fitness
                if f_trial > f_old:
                    pop[i] = trial

        # Return the best individual and its fitness
        best_idx = np.argmin(
            [f(self.makeSpan(p), self.totalCost(p)) for p in pop])
        best = pop[best_idx]
        best_fitness = f(self.makeSpan(best), self.totalCost(best))
        return best, best_fitness

    # print node allocated to particular task
    def TaskAllocation(self, chrom):
        print('************************Task Allocation**********************')
        for i in range(len(chrom)):
            print('Task_'+str(i+1)+' allocated to Node_'+str(chrom[i]+1))

    # print task allocated to particular node
    def GroupbyNodeAllocation(self, chrom):
        tempLst = []
        print('*********************GroupBy Node Allocation******************')
        for tem in range(self.TotalNode):
            tsk = [i for i, val in enumerate(chrom) if val == tem]
            tempLst.append(tsk)

        for i in range(len(tempLst)):
            temp = 'Node_'+str(i+1)+': '
            for j in tempLst[i]:
                temp += 'Task_'+str(j+1)+', '
            print(temp)



print('*************Resource Allocation in Vehicular Fog Computing*********************')
cloudN = int(input('Enter the Cloud Node required:'))
fogN = int(input('Enter the Fog Node required:'))
x = int(input('Enter the Task required:'))
Simulation(x, cloudN, fogN)
