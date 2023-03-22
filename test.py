import numpy as np


def differential_evolution(objective_func, bounds, pop, mutation=0.5, crossover=0.7, maxiter=1000, tol=1e-7):
    """
    Implements Differential Evolution optimization algorithm.

    Parameters:
    objective_func: function
        The objective function to be optimized.
    bounds: list of tuples
        The bounds for the input variables. Each tuple should contain the lower and upper bound of the corresponding variable.
    pop: list of lists
        The initial population. Each inner list should represent an individual in the population.
    mutation: float, optional (default=0.5)
        The mutation constant.
    crossover: float, optional (default=0.7)
        The crossover probability.
    maxiter: int, optional (default=1000)
        The maximum number of iterations.
    tol: float, optional (default=1e-7)
        The tolerance for convergence.

    Returns:
    best_solution: ndarray
        The best solution found by the algorithm.
    best_fitness: float
        The fitness value of the best solution.
    """
    popsize = len(pop)
    dim = len(bounds)
    min_b, max_b = np.asarray(bounds).T
    print(min_b,max_b)
    diff = np.fabs(max_b - min_b)
    print(diff)
    pop_denorm = [min_b + p * diff for p in pop]
    print(pop_denorm)
    fitness = np.asarray([objective_func(p) for p in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    best_fitness = fitness[best_idx]

    for i in range(maxiter):
        for j in range(popsize):
            idxs = list(range(popsize))
            idxs.remove(j)
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            print(a,b,c)
            mutant = np.clip(a + mutation * (b - c), 0, 1)
            cross_points = np.random.rand(dim) < crossover
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = objective_func(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                pop_denorm[j] = trial_denorm
                if f < best_fitness:
                    best_fitness = f
                    best = trial_denorm
        if np.std(fitness) < tol:
            break

    return best, best_fitness


def objective_function(x):
    return np.sum(x**2)


bounds = [(0, 13)]*10
print(bounds)
# pop = np.random.rand(20, 10).tolist()
# population_list
population_list=[]
for i in range(100):
    #print('i=',i)
    nxm_random_num=list(np.random.permutation(40)) # generate a random permutation of 0 to num_job*num_mc-1
    population_list.append(nxm_random_num) # add to the population_list
    for j in range(40):
        population_list[i][j]=population_list[i][j]%13 # convert to job number format, every job appears m times
# population_list
print(population_list)

best_solution, best_fitness = differential_evolution(
    objective_function, bounds, population_list)
print("Best solution: ", best_solution)
print("Best fitness: ", best_fitness)
