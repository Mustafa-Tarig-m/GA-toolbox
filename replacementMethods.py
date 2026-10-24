def generational_replacement(
    population,
    offspring,
    population_size=None,
    **_,
):
    size = len(population) if population_size is None else population_size
    return list(offspring[:size])


def elitist_replacement(
    population,
    offspring,
    population_size=None,
    **_,
):
    size = len(population) if population_size is None else population_size
    candidates = list(population) + list(offspring)
    candidates.sort(
        key=lambda chromosome: (
            chromosome.feasible is True,
            chromosome.fitness if chromosome.fitness is not None else float("-inf"),
        ),
        reverse=True,
    )
    return candidates[:size]