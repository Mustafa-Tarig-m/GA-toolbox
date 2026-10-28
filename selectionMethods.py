import random
def buildPropabilites(sortedPop)->dict:
    dictProp = {}
    fullRank = len(sortedPop)
    n_plus = 1.3 # could be changed
    n_minus = 0.7 
    for  i,indiv in enumerate(sortedPop):
        rank = i+1 
        prop = (n_minus + ((n_plus-n_minus)*((rank-1)/(fullRank-1))))/fullRank
        dictProp[indiv]= prop
    return dictProp

def rankSelection(population):
    sortedPop = population.sort(reversed= False , key = lambda item:item.fitness)
    propabilities =  buildPropabilites(sortedPop)
    r = random.random()

    cumFit = 0 
    for indi in population:
        cumFit = cumFit + propabilities[indi]
        if cumFit >= r :
            return indi
    return population[-1]
def TorunementSelection(population, k = 2):
    participints = random.sample(range(0,len(population)), 2)
    if(population[participints[0]]<= population[participints[1]]):
        return population[participints[1]]
    else: 
        population[participints[0]]