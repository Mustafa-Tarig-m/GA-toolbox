"""Run run_ga() on the OR-Tools CVRP example data."""

import random

import numpy as np

from CVRP import CVRPChromosome, CVRPInstance, CVRPProblem, print_solution
from GA import run_ga


DISTANCE_MATRIX = (
    (0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662),
    (548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210),
    (776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754),
    (696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358),
    (582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244),
    (274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708),
    (502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480),
    (194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856),
    (308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514),
    (194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468),
    (536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354),
    (502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844),
    (388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730),
    (354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536),
    (468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194),
    (776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798),
    (662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0),
)


class MatrixCVRPProblem(CVRPProblem):
    def distance_between(self, first_city: int, second_city: int) -> float:
        return float(DISTANCE_MATRIX[first_city][second_city])


def create_feasible_chromosome() -> CVRPChromosome:
    """Shuffle four known capacity-feasible routes from the OR-Tools data."""
    routes = [
        [7, 4, 3, 1],
        [8, 6, 5, 2],
        [15, 13, 11, 9],
        [16, 14, 12, 10],
    ]
    for route in routes:
        random.shuffle(route)
    return CVRPChromosome([0, *(city for route in routes for city in (*route, 0))])


def main() -> None:
    random.seed(7)
    np.random.seed(7)

    instance = CVRPInstance(
        vehicle_count=4,
        vehicle_capacity=15,
        demands=(0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8),
        locations=tuple((0.0, 0.0) for _ in DISTANCE_MATRIX),
    )
    problem = MatrixCVRPProblem(instance)

    best = run_ga(
        gene_type=CVRPChromosome,
        fitness_function=problem.fitness,
        feasibility_function=problem.feasibility,
        initializer=create_feasible_chromosome,
        configuration={
            "population_size": 200,
            "generations": 500,
            "mutation_rate": 0.25,
            "crossover_rate": 0.9,
            "crossover_method": problem.crossover,
            "mutation_method": problem.mutate,
        },
    )

    print_solution(problem, best)


if __name__ == "__main__":
    main()
