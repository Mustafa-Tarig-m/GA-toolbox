from collections.abc import Callable
from selectionMethods import rankSelection, TorunementSelection
import numpy as np

class Chromosome:
    def __init__(self, genes:list | None = None):
        self.genes = genes
        self.fitness: float | None = None
        self.feasible: bool | None = None

    def get_genes(self) -> np.ndarray:
        if self.genes is None:
            raise ValueError("Chromosome genes are not initialized.")
        return np.array(self.genes, dtype=np.uint8)

    def set_genes(self, genes: list):
        self.genes = genes

class BinaryChromosome(Chromosome):
    def __init__(self, genes: list):
        if not all(g in (0, 1, True, False) for g in genes):
            raise ValueError(
                "BinaryChromosome genes must strictly contain only 0s and 1s."
)
        super().__init__(genes)
    def get_genes(self):
        super().get_genes() 
        return np.array(self.genes, dtype=np.uint8)

class IntegerChromosome(Chromosome):
    def __init__(self, genes: list):
        if not all(isinstance(g, int) for g in genes):
            raise ValueError(
                "IntegerChromosome genes must strictly contain only integers."
            )
        super().__init__(genes)
    def get_genes(self): 
        super().get_genes()
        return np.array(self.genes, dtype=np.int64)

class FloatChromosome(Chromosome):
    def __init__(self, genes: list):
        if not all(isinstance(g, float) for g in genes):
            raise ValueError(
                "FloatChromosome genes must strictly contain only floats."
            )
        super().__init__(genes)
    def get_genes(self): 
        super().get_genes()
        return np.array(self.genes, dtype=np.float64)
    
class GA:
    def __init__(self,
                 gene_type: type[Chromosome],
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

        self.gene_type = gene_type
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

    def configure(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"GA class has no attribute '{key}'")

    def init_population(self) -> list[Chromosome]:
        population = []
        for _ in range(self.population_size):
            if self.gene_type == BinaryChromosome:
                genes = np.random.randint(0, 2, size=10).tolist() 
            elif self.gene_type == IntegerChromosome:
                genes = np.random.randint(0, 100, size=10).tolist() 
            elif self.gene_type == FloatChromosome:
                genes = np.random.uniform(0.0, 1.0, size=10).tolist()
            else:
                raise ValueError("Unsupported gene type")
            chromosome = self.gene_type(genes)
            population.append(chromosome)
        return population

    def evaluate_fitness(self, population: list[Chromosome]) -> list[Chromosome]:
        ...

    def check_feasibility(self, chromosome):
        ...

    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        fitnessScores = self.evaluate_fitness(population)
        matingPool = []
        for i in range(self.population_size): # mating pool is the same as the population size 
            selectedChrom = self.selection_method(population,fitnessScores)
            matingPool.append(selectedChrom)
        return matingPool

    def crossover(self, parents) -> list[Chromosome]:
        ...

    def mutate(self, offspring_candidates):
        ...

    def replacement(self, mutated_offsprings: list[Chromosome]):
        ...

    def run(self):
        population = self.init_population()
        for _ in range(self.generations):
            evaluated_population = self.evaluate_fitness(population)
            parents = self.select(evaluated_population)
            offspring_candidates = self.crossover(parents)
            mutated_offsprings = self.mutate(offspring_candidates)
            population = self.replacement(mutated_offsprings)



