import random
import numpy as np


def differential_evolution(pop_size, generations, f, cr, bounds, NoOfTask, makeSpan, totalCost):
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
                temp = random.uniform(0, 1)
                if temp < cr:
                    trial[j] = mutant[j]
                    # print('in',temp,j)
            # print('trail:',trial)

            # Clip trial vector to bounds
            trial = np.clip(trial, bounds[0], bounds[1])
            # print('trail:',trial)

            # Evaluate fitness of trial vector
            f_trial = f(makeSpan(trial), totalCost(trial))
            f_old = f(makeSpan(pop[i]), totalCost(pop[i]))

            # Replace individual with trial vector if it has better fitness
            if f_trial > f_old:
                pop[i] = trial

    # Return the best individual and its fitness
    best_idx = np.argmin([f(makeSpan(p), totalCost(p)) for p in pop])
    best = pop[best_idx]
    best_fitness = f(makeSpan(best), totalCost(best))
    return best, best_fitness
