"""Generic genetic algorithm implementation and chromosome types."""

from collections.abc import Callable, Mapping
import random

import numpy as np

from mutationMethods import flip_bit, non_uniform_mutation, uniform_mutation
from replacementMethods import elitist_replacement


class Chromosome:
    def __init__(self):
        self.genes = None
        self.fitness: float | None = None
        self.feasible: bool | None = None

    def get_genes(self) -> np.ndarray:
        if self.genes is None:
            raise ValueError("Chromosome genes are not initialized.")
        return np.array(self.genes)


class BinaryChromosome(Chromosome):
    def __init__(self, genes: list):
        super().__init__()
        self.genes = [int(gene) for gene in genes]
        if not all(gene in (0, 1) for gene in self.genes):
            raise ValueError("BinaryChromosome genes must contain only 0s and 1s.")

    def get_genes(self) -> np.ndarray:
        return np.array(self.genes, dtype=np.uint8)


class IntegerChromosome(Chromosome):
    def __init__(self, genes: list, min_val: int = 0, max_val: int = 100):
        super().__init__()
        self.genes = [int(gene) for gene in genes]
        self.min_val = min_val
        self.max_val = max_val

    def get_genes(self) -> np.ndarray:
        return np.array(self.genes, dtype=np.int64)


class FloatChromosome(Chromosome):
    def __init__(self, genes: list, min_val: float = 0.0, max_val: float = 1.0):
        super().__init__()
        self.genes = [float(gene) for gene in genes]
        self.min_val = min_val
        self.max_val = max_val

    def get_genes(self) -> np.ndarray:
        return np.array(self.genes, dtype=np.float64)


class GA:
    def __init__(
        self,
        gene_type: type[Chromosome],
        fitness_function: Callable,
        feasibility_function: Callable,
        population_size: int = 100,
        generations: int = 200,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        selection_method: Callable | None = None,
        crossover_method: Callable | None = None,
        mutation_method: Callable | None = None,
        replacement_method: Callable | None = None,
        initializer: Callable[[], Chromosome] | None = None,
    ):
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
        self.replacement_method = replacement_method or elitist_replacement
        self.initializer = initializer
        self._validate_settings()

    def _validate_settings(self) -> None:
        if self.population_size < 1:
            raise ValueError("population_size must be at least 1.")
        if self.generations < 0:
            raise ValueError("generations cannot be negative.")
        for name, rate in (
            ("mutation_rate", self.mutation_rate),
            ("crossover_rate", self.crossover_rate),
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

    def configure(self, **kwargs) -> "GA":
        """Update algorithm settings and return this instance.

        This is useful when an application wants to expose a small set of
        algorithm options without rebuilding the problem object.
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"GA class has no attribute '{key}'")
            setattr(self, key, value)
        self._validate_settings()
        return self

    def init_population(self) -> list[Chromosome]:
        if self.initializer is not None:
            return [self.initializer() for _ in range(self.population_size)]

        population = []
        for _ in range(self.population_size):
            if self.gene_type is BinaryChromosome:
                genes = np.random.randint(0, 2, size=10).tolist()
                population.append(BinaryChromosome(genes))
            elif self.gene_type is IntegerChromosome:
                genes = np.random.randint(0, 100, size=10).tolist()
                population.append(IntegerChromosome(genes))
            elif self.gene_type is FloatChromosome:
                genes = np.random.uniform(0.0, 1.0, size=10).tolist()
                population.append(FloatChromosome(genes))
            else:
                raise ValueError("An initializer is required for this chromosome type.")
        return population

    def evaluate_fitness(self, population: list[Chromosome]) -> list[Chromosome]:
        for chromosome in population:
            chromosome.feasible = bool(self.feasibility_function(chromosome))
            chromosome.fitness = (
                float(self.fitness_function(chromosome))
                if chromosome.feasible
                else float("-inf")
            )
        return population

    def select(self, population: list[Chromosome]) -> list[Chromosome]:
        if self.selection_method is None:
            return [
                max(
                    random.sample(population, min(2, len(population))),
                    key=lambda chromosome: (
                        chromosome.feasible is True,
                        chromosome.fitness
                        if chromosome.fitness is not None
                        else float("-inf"),
                    ),
                )
                for _ in range(self.population_size)
            ]
        return [
            self.selection_method(population) for _ in range(self.population_size)
        ]

    def crossover(self, parents: list[Chromosome]) -> list[Chromosome]:
        if self.crossover_method is None:
            return list(parents)
        return self.crossover_method(parents, self.crossover_rate)

    def mutate(
        self,
        offspring: list[Chromosome],
        current_generation: int,
    ) -> list[Chromosome]:
        if self.mutation_method is not None:
            return [
                self.mutation_method(
                    chromosome,
                    self.mutation_rate,
                    current_generation,
                    self.generations,
                )
                for chromosome in offspring
            ]

        mutated = []
        for chromosome in offspring:
            genes = chromosome.get_genes()
            if self.gene_type is BinaryChromosome:
                genes = flip_bit(self.mutation_rate, genes)
            elif self.gene_type is IntegerChromosome:
                genes = uniform_mutation(
                    self.mutation_rate, genes, chromosome.max_val, chromosome.min_val
                )
            elif self.gene_type is FloatChromosome:
                genes = non_uniform_mutation(
                    self.mutation_rate,
                    genes,
                    chromosome.max_val,
                    chromosome.min_val,
                    current_generation,
                    self.generations,
                )
            mutated.append(self.gene_type(genes.tolist()))
        return mutated

    def replacement(
        self,
        population: list[Chromosome],
        offspring: list[Chromosome],
    ) -> list[Chromosome]:
        return self.replacement_method(
            population,
            offspring,
            population_size=self.population_size,
        )

    def run(self) -> Chromosome:
        population = self.evaluate_fitness(self.init_population())
        best = max(population, key=lambda chromosome: chromosome.fitness)

        for generation in range(self.generations):
            parents = self.select(population)
            offspring = self.mutate(self.crossover(parents), generation)
            offspring = self.evaluate_fitness(offspring)
            population = self.replacement(population, offspring)
            population = self.evaluate_fitness(population)
            candidate = max(population, key=lambda chromosome: chromosome.fitness)
            if candidate.fitness > best.fitness:
                best = candidate

        return best

#wrapper function thats used to run the GA class
def run_ga(
    *,
    gene_type: type[Chromosome],
    fitness_function: Callable,
    feasibility_function: Callable,
    initializer: Callable[[], Chromosome],
    configuration: Mapping[str, object] | None = None,
) -> Chromosome:

    options = configuration or {}

    for name in ["gene_type", "fitness_function", "feasibility_function", "initializer"]:
        if name in options:
            raise ValueError(f"{name} cannot be set in configuration.")

    ga = GA(
        gene_type=gene_type,
        fitness_function=fitness_function,
        feasibility_function=feasibility_function,
        initializer=initializer,
        **options,
    )

    return ga.run()