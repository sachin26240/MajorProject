import random
import numpy as np
import pandas as pd
import time
from statistics import mean


def PSO(pop_size, generations, f, NoOfTask,TotalNode, makeSpan, totalCost):
    # task = [1,0,7,9,5,4,3,8,6,1]
    def List2Matrix(task):
        lst2mat = []
        for Ni in range(TotalNode):
            temp = []
            for Ti in range(NoOfTask):
                temp.append(0)
            lst2mat.append(temp)

        for Ti in range(NoOfTask):
            lst2mat[task[Ti]][Ti] = 1
        return lst2mat

    def Matrix2List(task):
        mat2lst = []
        for Ti in range(NoOfTask):
            for Ni in range(TotalNode):
                if task[Ni][Ti] == 1:
                    mat2lst.append(Ni)
        return mat2lst

    def velocityMatix(rang):
        mat = []
        for Ni in range(TotalNode):
            temp = []
            for Ti in range(NoOfTask):
                temp.append(round(random.uniform(-rang, rang), 8))
            mat.append(temp)
        return mat

    num_task = NoOfTask
    num_vm = TotalNode
    num_population = pop_size
    num_iteration = generations
    c1 = 1.5
    c2 = 1.5
    w = 0.6
    Vmax = 28

    population_list = []
    for i in range(num_population):
        # print('i=',i)
        # generate a random permutation of 0 to num_job*num_mc-1
        nxm_random_num = list(np.random.permutation(num_task))
        population_list.append(nxm_random_num)  # add to the population_list
        for j in range(num_task):
            # convert to job number format, every job appears m times
            population_list[i][j] = population_list[i][j] % num_vm
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
    gbest = population_list[random.randint(0, num_population-1)]
    # loop
    for i in range(num_iteration):
        # print('iteration:',i)
        for particle in range(num_population):
            mat2lst = Matrix2List(lst2mat[particle])
            # print(mat2lst)
            tc = totalCost(mat2lst)
            ms = makeSpan(mat2lst)
            fx = f(ms, tc)
            pb = f(
                makeSpan(pbest[particle]), totalCost(pbest[particle]))
            if fx > pb:
                pbest[particle] = mat2lst
                pb = fx

            gb = f(makeSpan(gbest), totalCost(gbest))
            # print('before gbest',pb,gb)
            if pb > gb:
                gbest = pbest[particle]
                gb = pb

        # update velocity
        for particle in range(num_population):
            pbest2Mat = List2Matrix(pbest[particle])
            gbest2Mat = List2Matrix(gbest)
            for Ni in range(TotalNode):
                # temp = []
                for Ti in range(NoOfTask):
                    vel[particle][Ni][Ti] = (w*vel[particle][Ni][Ti])+((c1*0.4)*(
                        pbest2Mat[Ni][Ti]-lst2mat[particle][Ni][Ti]))+((c2*0.6)*(gbest2Mat[Ni][Ti]-lst2mat[particle][Ni][Ti]))

            # update position
            velTranspose = [[row[i] for row in vel[particle]]
                            for i in range(len(vel[particle][0]))]
            for Ti in range(NoOfTask):
                for Ni in range(TotalNode):
                    if Ni == velTranspose[Ti].index(max(velTranspose[Ti])):
                        lst2mat[particle][Ni][Ti] = 1
                    else:
                        lst2mat[particle][Ni][Ti] = 0
            # print(lst2mat)

    return gbest,f(makeSpan(gbest), totalCost(gbest))