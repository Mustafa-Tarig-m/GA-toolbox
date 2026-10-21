from collections.abc import Callable
from selectionMethods import rankSelection, TorunementSelection

class Chromosome:
    def __init__(self, chromosome: str, type):
        pass

    
class GA:
    def __init__(self,
                 fitness_function: Callable,
                 feasibility_function: Callable,
                 population_size:int = 100, 
                 generations:int = 200, 
                 mutation_rate:float = 0.1, 
                 crossover_rate:float = 0.8, 
                 selection_method: Callable | None = None,
                 crossover_method: Callable | None = None,
                 mutation_method: Callable | None = None,
                 replacement_method: Callable | None = None):

        self.fitness_function = fitness_function
        self.feasibility_function = feasibility_function
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutation_method = mutation_method
        self.replacement_method = replacement_method

    def init_population(self) -> list:
        ...

    def evaluate_fitness(self, population: list) -> dict:
        ...

    def check_feasibility(self, chromosome):
        ...

    def select(self, population: list):
        fitnessScores = evaluate_fitness(population)
        matingPool = []
        for i in range(self.population_size): # mating pool is the same as the population size 
            selectedChrom = self.selection_method(population,fitnessScores)
            matingPool.append(selectedChrom)
        return matingPool

    def crossover(self, p1, p2):
        ...

    def mutate(self, chromosome):
        ...

    def replacement(self):
        ...

    def run(self):
        ...