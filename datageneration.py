import random
import math
from statistics import mean
import time
import pandas as pd
# t1 = time.perf_counter()

def dataGeneration(path,numoftask,numofcloudnode,numoffognode):
    Excel_File= pd.ExcelWriter(path, engine = 'openpyxl')

    NoOfTask = numoftask
    NoOfCloudNode = numofcloudnode
    NoOfFogNode = numoffognode
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
    # print(ExecutionTable_DF)
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

dataGeneration('Excel File/task160.xlsx',160,3,10)

# #minmakespan
# lengthSum = 0
# cpuRateSum = 0
# for x in range(NoOfTask):
#   lengthSum+=TaskDetails[x][0]

# for x in range(TotalNode):
#   cpuRateSum+=NodeDetails[x][0]

# minmakespan = round((lengthSum*(10**3))/cpuRateSum , 4 )
# # minmakespan

# #minTotalCost
# costListTranspose =[[row[i] for row in costList] for i in range(len(costList[0]))]
# minTotalcost = []
# for i in range(NoOfTask):
#   minTotalcost.append(min(costListTranspose[i]))
# minTotalcost = sum(minTotalcost)
# # minTotalcost

# #utility function
# def utilityFunction(makespan,totalcost,minTotalcost=minTotalcost,minmakespan=minmakespan,alpha = 0.5):
#   # alpha = 0.5
#   x= (alpha*(minmakespan/makespan)) + ((1-alpha)*(minTotalcost/totalcost))
#   return x


# def totalCost(chrom):
#   sum = 0
#   for x in range(NoOfTask):
#     sum+=costList[chrom[x]][x]
#   return round(sum,4)

# # x=[1,6,5,2,1,6,1,5,9,6]

# def makeSpan(chrom):
#   # NiTasks = []
#   mspan = []
#   for tem in range(TotalNode):
#     val = 0
#     tsk = [i for i,val in enumerate(chrom) if val==tem]
#     # NiTasks.append([i for i,val in enumerate(x) if val==tem])
#     for sac in tsk:
#       val+=eTimeList[tem][sac]
#     mspan.append(val)
#   # print(NiTasks)
#   return max(mspan)