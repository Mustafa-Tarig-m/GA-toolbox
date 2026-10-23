import random

def flip_bit(mutationRate, arr):
    carr = arr.copy()
    for i in range(len(arr)):
        if random.uniform(0, 1) < mutationRate:
            carr[i] = 1 - carr[i]
    return carr

def uniform_mutation(mutationRate, arr, ubound, lbound):
    carr = arr.copy()
    for i in range(len(carr)):
        if random.uniform(0, 1) < mutationRate:
            delta = 0
            deltaL = carr[i] - lbound
            deltaU = ubound - carr[i]
            if random.uniform(0, 1) <= 0.5:
                delta = deltaL
            else:
                delta = deltaU
            if delta == deltaL:
                carr[i] = carr[i] - random.uniform(0, delta)
            else:
                carr[i] = carr[i] + random.uniform(0, delta)
    return carr

def non_uniform_mutation(mutationRate, arr, ubound, lbound, current_generation, max_generations):
    carr = arr.copy()
    for i in range(len(carr)):
        if random.uniform(0, 1) < mutationRate:
            delta = 0
            deltaL = carr[i] - lbound
            deltaU = ubound - carr[i]
            if random.uniform(0, 1) <= 0.5:
                delta = deltaL
            else:
                delta = deltaU
            r = random.uniform(0, 1)
            b = 5 # could be changed
            mutation_value = delta * (1 - r ** ((1 - current_generation / max_generations) ** b))
            if delta == deltaL:
                carr[i] = carr[i] - mutation_value
            else:
                carr[i] = carr[i] + mutation_value
    return carr