from copy import copy
import random
from collections.abc import Sequence


def to_gene_list(chromosome) -> list:
    """Return a chromosome's genes without changing the source chromosome."""
    if not hasattr(chromosome, "genes") or chromosome.genes is None:
        raise ValueError("Each parent must have initialized genes.")
    return list(chromosome.genes)


def make_child(parent, genes):
    """Create a child while retaining the parent's chromosome subclass."""
    child = copy(parent)
    child.genes = list(genes)
    child.fitness = None
    child.feasible = None
    return child


def parent_pairs(parents):
    if len(parents) < 2:
        raise ValueError("At least two parents are required for crossover.")
    if len(parents) % 2:
        raise ValueError("The number of parents must be even.")
    return zip(parents[::2], parents[1::2])


def multi_point_crossover(
    parents: Sequence,
    num_points: int = 2,
    crossover_rate: float = 1.0,
) -> list:
    """Apply multi-point crossover to adjacent pairs of chromosomes.

    Each pair produces two children. ``num_points`` distinct cut positions are
    selected between genes, and segments alternate between the two parents.
    The operator works with binary, integer, and floating-point chromosomes.
    When crossover is skipped, copies of the two parents are returned.
    """
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be between 0 and 1.")
    offspring = []

    for first_parent, second_parent in parent_pairs(parents):
        first_genes = to_gene_list(first_parent)
        second_genes = to_gene_list(second_parent)
        if len(first_genes) != len(second_genes):
            raise ValueError("Parents must have the same number of genes.")
        length = len(first_genes)
        if num_points < 1 or num_points >= length:
            raise ValueError(
                "num_points must be at least 1 and smaller than the chromosome length."
            )

        if random.random() >= crossover_rate:
            offspring.extend(
                [make_child(first_parent, first_genes),
                 make_child(second_parent, second_genes)]
            )
            continue

        cuts = sorted(random.sample(range(1, length), num_points))
        boundaries = [0, *cuts, length]
        child_one = []
        child_two = []
        for segment, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if segment % 2 == 0:
                child_one.extend(first_genes[start:end])
                child_two.extend(second_genes[start:end])
            else:
                child_one.extend(second_genes[start:end])
                child_two.extend(first_genes[start:end])
        offspring.extend(
            [make_child(first_parent, child_one),
             make_child(second_parent, child_two)]
        )
    return offspring


def order_crossover(parents: Sequence, crossover_rate: float = 1.0) -> list:
    """Apply order crossover (OX) to adjacent pairs of permutation chromosomes.

    OX copies a randomly selected segment from each parent and fills the
    remaining positions in the order encountered in the other parent. Parent
    genes must be permutations of the same unique values.
    """
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be between 0 and 1.")
    offspring = []

    for first_parent, second_parent in parent_pairs(parents):
        first_genes = to_gene_list(first_parent)
        second_genes = to_gene_list(second_parent)
        if len(first_genes) != len(second_genes):
            raise ValueError("Parents must have the same number of genes.")
        if (
            len(set(first_genes)) != len(first_genes)
            or len(set(second_genes)) != len(second_genes)
            or set(first_genes) != set(second_genes)
        ):
            raise ValueError(
                "Order crossover requires parents with the same unique genes."
            )
        if len(first_genes) < 2:
            raise ValueError(
                "Order crossover requires chromosomes with at least two genes."
            )

        if random.random() >= crossover_rate:
            offspring.extend(
                [make_child(first_parent, first_genes),
                 make_child(second_parent, second_genes)]
            )
            continue

        start, end = sorted(random.sample(range(len(first_genes)), 2))
        end += 1

        def build_child(segment_parent, fill_parent):
            child = [None] * len(segment_parent)
            child[start:end] = segment_parent[start:end]
            remaining = [
                gene for gene in fill_parent if gene not in child[start:end]
            ]
            fill_positions = list(range(end, len(child))) + list(range(start))
            for position, gene in zip(fill_positions, remaining):
                child[position] = gene
            return child

        offspring.extend(
            [make_child(first_parent, build_child(first_genes, second_genes)),
             make_child(second_parent, build_child(second_genes, first_genes))]
        )
    return offspring

