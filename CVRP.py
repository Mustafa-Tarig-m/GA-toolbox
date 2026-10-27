
"""Capacitated vehicle routing problem (CVRP) model for the GA toolbox."""

from dataclasses import dataclass
from collections.abc import Sequence
from GA import Chromosome

import math
import random
import numpy as np

from GA import run_ga

Point = tuple[float, float]


@dataclass
class CVRPInstance:
    """ CVRP

    City ``0`` is the depot.
    Vehicles start and finish at the depot.
    """

    vehicle_count: int
    vehicle_capacity: int
    demands: tuple[int, ...]
    locations: tuple[Point, ...]

    @property
    def city_count(self) -> int:
        return len(self.demands)

    def __post_init__(self) -> None:
        if self.vehicle_count < 1:
            raise ValueError("vehicle_count must be at least 1.")
        if self.vehicle_capacity < 1:
            raise ValueError("vehicle_capacity must be at least 1.")
        if len(self.demands) < 2:
            raise ValueError("At least one customer and the depot are required.")
        if len(self.demands) != len(self.locations):
            raise ValueError("demands and locations must have the same length.")
        if self.demands[0] != 0:
            raise ValueError("The depot (city 0) must have demand 0.")
        if self.locations[0] != (0.0, 0.0):
            raise ValueError("The depot (city 0) must be located at (0, 0).")

        for city, demand in enumerate(self.demands):
            if not isinstance(demand, int) or isinstance(demand, bool):
                raise ValueError(f"Demand for city {city} must be an integer.")
            if demand < 0:
                raise ValueError(f"Demand for city {city} cannot be negative.")
            if demand > self.vehicle_capacity:
                raise ValueError(
                    f"Demand for city {city} exceeds vehicle capacity."
                )

        for city, location in enumerate(self.locations):
            if len(location) != 2 or not all(
                isinstance(coordinate, (int, float))
                and not isinstance(coordinate, bool)
                and math.isfinite(coordinate)
                for coordinate in location
            ):
                raise ValueError(
                    f"Location for city {city} must contain two finite numbers."
                )


@dataclass
class VehicleRoute:
    cities: tuple[int, ...]
    load: int
    distance: float


class CVRPChromosome(Chromosome):

    def __init__(self, genes: Sequence[int]):
        super().__init__()
        self.genes = [int(gene) for gene in genes]

    def get_genes(self) -> np.ndarray:
        if self.genes is None:
            raise ValueError("chromosome genes are not initialized.")
        return np.array(self.genes, dtype=np.int64)


class CVRPProblem:
    """CVRP operations used by chromosome initialization and the GA."""

    def __init__(self, instance: CVRPInstance):
        self.instance = instance

    def routes_from_genes(self, genes: Sequence[int]) -> tuple[tuple[int, ...], ...]:
        """Split depot-delimited genes into exactly one route per vehicle."""
        if not genes or genes[0] != 0:
            raise ValueError("A chromosome must start at the depot (0).")
        routes: list[tuple[int, ...]] = []
        current = [0]
        for city in genes[1:]:
            if not isinstance(city, int) or city < 0:
                raise ValueError("Chromosome genes must be non-negative integers.")
            if city >= self.instance.city_count:
                raise ValueError(f"City index {city} is outside the problem.")
            current.append(city)
            if city == 0:
                routes.append(tuple(current))
                current = [0]
        if current != [0]:
            raise ValueError("A chromosome must end at the depot (0).")
        if len(routes) != self.instance.vehicle_count:
            raise ValueError(
                f"Expected {self.instance.vehicle_count} routes, got {len(routes)}."
            )
        return tuple(routes)

    def route_load(self, route: Sequence[int]) -> int:
        return sum(self.instance.demands[city] for city in route if city != 0)

    def distance_between(self, first_city: int, second_city: int) -> float:
        first = self.instance.locations[first_city]
        second = self.instance.locations[second_city]
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def route_distance(self, route: Sequence[int]) -> float:
        return sum(
            self.distance_between(first, second)
            for first, second in zip(route, route[1:])
        )

    def validate_chromosome(
        self, chromosome: CVRPChromosome | Sequence[int]
    ) -> bool:
        """Return whether every customer is visited once within capacity."""
        genes = chromosome.genes if isinstance(chromosome, Chromosome) else chromosome
        try:
            routes = self.routes_from_genes(genes)
        except (TypeError, ValueError):
            return False
        visited = [city for route in routes for city in route[1:-1]]
        return (
            sorted(visited) == list(range(1, self.instance.city_count))
            and all(self.route_load(route) <= self.instance.vehicle_capacity
                    for route in routes)
        )

    def total_distance(self, chromosome: CVRPChromosome | Sequence[int]) -> float:
        #uses the Manhattan distance to calculate the total distance 
        genes = chromosome.genes if isinstance(chromosome, Chromosome) else chromosome
        if not self.validate_chromosome(genes):
            raise ValueError("The chromosome is not a feasible CVRP solution.")
        return sum(
            self.route_distance(route)
            for route in self.routes_from_genes(genes)
        )

    def fitness(self, chromosome: CVRPChromosome) -> float:
    
        return -self.total_distance(chromosome)

    def feasibility(self, chromosome: CVRPChromosome) -> bool:
        return self.validate_chromosome(chromosome)

    def random_chromosome(self) -> CVRPChromosome:
    
        customers = list(range(1, self.instance.city_count))
        random.shuffle(customers)
        routes: list[list[int]] = [[] for _ in range(self.instance.vehicle_count)]
        loads = [0] * self.instance.vehicle_count
        for city in customers:
            possible = [
                vehicle for vehicle in range(self.instance.vehicle_count)
                if loads[vehicle] + self.instance.demands[city]
                <= self.instance.vehicle_capacity
            ]
            if not possible:
                raise ValueError(
                    "The entered vehicles and capacity cannot accommodate "
                    "the generated customer assignment."
                )
            vehicle = random.choice(possible)
            routes[vehicle].append(city)
            loads[vehicle] += self.instance.demands[city]

        genes = [0]
        for route in routes:
            genes.extend(route)
            genes.append(0)
        return CVRPChromosome(genes)

    def crossover(
        self,
        parents: list[CVRPChromosome],
        crossover_rate: float,
    ) -> list[CVRPChromosome]:
        offspring: list[CVRPChromosome] = []
        for first, second in zip(parents[::2], parents[1::2]):
            first_routes = self.routes_from_genes(first.genes)
            second_routes = self.routes_from_genes(second.genes)
            first_customers = [
                city for route in first_routes for city in route if city != 0
            ]
            second_customers = [
                city for route in second_routes for city in route if city != 0
            ]

            if random.random() >= crossover_rate:
                children = [first, second]
            elif len(first_customers) < 2:
                children = [first, second]
            else:
                start, end = sorted(
                    random.sample(range(len(first_customers)), 2)
                )

                def order_child(segment, fill):
                    child = [None] * len(segment)
                    child[start:end] = segment[start:end]
                    remaining = [
                        city for city in fill if city not in child[start:end]
                    ]
                    positions = list(range(end, len(child))) + list(range(start))
                    for position, city in zip(positions, remaining):
                        child[position] = city
                    return child

                child_customers = [
                    order_child(first_customers, second_customers),
                    order_child(second_customers, first_customers),
                ]
                route_lengths = [
                    [len(route) - 2 for route in first_routes],
                    [len(route) - 2 for route in second_routes],
                ]
                children = []
                for customers, lengths, fallback in zip(
                    child_customers, route_lengths, [first, second]
                ):
                    genes = [0]
                    offset = 0
                    for length in lengths:
                        genes.extend(customers[offset:offset + length])
                        genes.append(0)
                        offset += length
                    child = CVRPChromosome(genes)
                    children.append(child if self.validate_chromosome(child) else fallback)

            offspring.extend(children)
        return offspring

    def mutate(
        self,
        chromosome: CVRPChromosome,
        mutation_rate: float,
        _generation: int,
        _generations: int,
    ) -> CVRPChromosome:
        if random.random() >= mutation_rate:
            return chromosome
        routes = [list(route) for route in self.routes_from_genes(chromosome.genes)]
        customer_positions = [
            (route_index, city_index)
            for route_index, route in enumerate(routes)
            for city_index in range(1, len(route) - 1)
        ]
        if len(customer_positions) < 2:
            return chromosome
        first_position, second_position = random.sample(customer_positions, 2)
        first_route, first_index = first_position
        second_route, second_index = second_position
        routes[first_route][first_index], routes[second_route][second_index] = (
            routes[second_route][second_index],
            routes[first_route][first_index],
        )
        genes = [city for route in routes for city in route]
        candidate = CVRPChromosome(genes)
        return candidate if self.validate_chromosome(candidate) else chromosome

    def report(self, chromosome: CVRPChromosome) -> tuple[VehicleRoute, ...]:
        if not self.validate_chromosome(chromosome):
            raise ValueError("Cannot report an infeasible CVRP chromosome.")
        return tuple(
            VehicleRoute(
                cities=route,
                load=self.route_load(route),
                distance=self.route_distance(route),
            )
            for route in self.routes_from_genes(chromosome.genes)
        )


def print_solution(problem: CVRPProblem, chromosome: CVRPChromosome) -> None:
    routes = problem.report(chromosome)

    for i, route in enumerate(routes, 1):
        cities = " -> ".join(str(city) for city in route.cities)

        print(f"Vehicle {i}: {cities}")
        print(f"  Load: {route.load}")
        print(f"  Distance: {route.distance}")

    total = problem.total_distance(chromosome)
    print(f"\nTotal distance: {total}")


GA_OPTIONS = {
    "population_size": 100,
    "generations": 200,
    "mutation_rate": 0.15,
    "crossover_rate": 0.9,
}


def read_integer(prompt: str, minimum: int = 0) -> int:
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if value < minimum:
            print(f"Please enter an integer greater than or equal to {minimum}.")
            continue
        return value


def read_demands(city_count: int) -> tuple[int, ...]:
    while True:
        try:
            demands = tuple(int(value) for value in input(
                f"Enter demands for {city_count} cities, separated by spaces: "
            ).split())
        except ValueError:
            print("Every demand must be an integer.")
            continue
        if len(demands) != city_count:
            print(f"Please enter exactly {city_count} demands.")
            continue
        return demands


def read_locations(city_count: int) -> tuple[Point, ...]:
    print("Enter one x y pair for each city, separated by spaces.")
    print("City 0 is the depot and must be entered as: 0 0")
    while True:
        locations: list[Point] = []
        valid = True
        for city in range(city_count):
            try:
                coordinates = tuple(float(value) for value in input(
                    f"Location for city {city}: "
                ).split())
            except ValueError:
                valid = False
                break
            if len(coordinates) != 2:
                valid = False
                break
            locations.append((coordinates[0], coordinates[1]))
        if valid and len(locations) == city_count:
            return tuple(locations)
        print("Each location must contain exactly two numeric coordinates.")


def read_cvrp_instance() -> CVRPInstance:
    vehicle_count = read_integer("Number of vehicles: ", minimum=1)
    vehicle_capacity = read_integer("Vehicle capacity: ", minimum=1)
    city_count = read_integer("Number of cities: ", minimum=2)
    while True:
        demands = read_demands(city_count)
        locations = read_locations(city_count)
        try:
            return CVRPInstance(
                vehicle_count=vehicle_count,
                vehicle_capacity=vehicle_capacity,
                demands=demands,
                locations=locations,
            )
        except ValueError as error:
            print(f"Invalid CVRP instance: {error}")
            print("Please enter the demands and locations again.")


def main() -> None:
    problem = CVRPProblem(read_cvrp_instance())
    chromosome = run_ga(
        gene_type=CVRPChromosome,
        fitness_function=problem.fitness,
        feasibility_function=problem.feasibility,
        initializer=problem.random_chromosome,
        configuration={
            **GA_OPTIONS,
            "crossover_method": problem.crossover,
            "mutation_method": problem.mutate,
        },
    )
    print_solution(problem, chromosome)


if __name__ == "__main__":
    main()
