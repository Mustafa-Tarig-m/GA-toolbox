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

def rankSelection(population,fitnessScores):
    sortedPop = population.sort(reversed= True , key = lambda item:fitnessScores[item])
    propabilities =  buildPropabilites(sortedPop)
    r = random.random()

    cumFit = 0 
    for indi in population:
        cumFit = cumFit + propabilities[indi]
        if cumFit >= r :
            return indi
    return population[-1]
def TorunementSelection(population,fitnessScores, k = 2):
    ...