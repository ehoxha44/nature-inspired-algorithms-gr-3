from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class GAConfig:

    population_size:  int   = 100
    num_generations:  int   = 200
    crossover_rate:   float = 0.85
    mutation_rate:    float = 0.20
    tournament_size:  int   = 5
    elitism_count:    int   = 10
    stagnation_limit: int   = 30
    seed: Optional[int]     = 42


class GeneticAlgorithm:

    _STAGNATION_BOOST = 3.0
    _MAX_MUTATION     = 0.80

    def __init__(self, config: GAConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)

    def run(
        self,
        chromosome_length: int,
        fitness_fn: Callable[[List[int]], float],
        seed_chromosomes: Optional[List[List[int]]] = None,
        verbose: bool = True,
        log_interval: int = 25,
    ) -> Tuple[List[int], float, List[float]]:
        cfg = self.config

        population = self._init_population(chromosome_length, seed_chromosomes)

        best_chromosome: List[int] = population[0][:]
        best_fitness: float = float("-inf")
        convergence: List[float] = []
        stagnant_gens: int = 0

        for gen in range(cfg.num_generations):

            scores = [fitness_fn(ch) for ch in population]

            gen_best_idx = max(range(len(scores)), key=lambda i: scores[i])
            if scores[gen_best_idx] > best_fitness:
                best_fitness   = scores[gen_best_idx]
                best_chromosome = population[gen_best_idx][:]
                stagnant_gens  = 0
            else:
                stagnant_gens += 1

            convergence.append(best_fitness)

            if stagnant_gens >= cfg.stagnation_limit:
                eff_mut = min(self._MAX_MUTATION,
                              cfg.mutation_rate * self._STAGNATION_BOOST)
            else:
                eff_mut = cfg.mutation_rate

            if verbose and gen % log_interval == 0:
                avg = sum(scores) / len(scores)
                boost_marker = " [boost]" if eff_mut > cfg.mutation_rate else ""
                print(
                    f"  Gen {gen:4d}/{cfg.num_generations} | "
                    f"best={best_fitness:8.1f} | "
                    f"gen_best={scores[gen_best_idx]:8.1f} | "
                    f"avg={avg:8.1f}{boost_marker}"
                )

            next_pop: List[List[int]] = []

            elite_idxs = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[: cfg.elitism_count]
            for idx in elite_idxs:
                next_pop.append(population[idx][:])

            while len(next_pop) < cfg.population_size:
                p1 = self._tournament_select(population, scores)
                p2 = self._tournament_select(population, scores)

                if self._rng.random() < cfg.crossover_rate:
                    c1, c2 = self._order_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]

                if self._rng.random() < eff_mut:
                    c1 = self._mutate(c1)
                if self._rng.random() < eff_mut:
                    c2 = self._mutate(c2)

                next_pop.append(c1)
                if len(next_pop) < cfg.population_size:
                    next_pop.append(c2)

            population = next_pop

        if verbose:
            print(f"\n  Finished. Best fitness = {best_fitness:.1f}")

        return best_chromosome, best_fitness, convergence

    def _init_population(
        self,
        length: int,
        seeds: Optional[List[List[int]]],
    ) -> List[List[int]]:
        population: List[List[int]] = []

        if seeds:
            for s in seeds[: self.config.population_size]:
                population.append(s[:])

        base = list(range(length))
        while len(population) < self.config.population_size:
            perm = base[:]
            self._rng.shuffle(perm)
            population.append(perm)

        return population

    def _tournament_select(
        self, population: List[List[int]], scores: List[float]
    ) -> List[int]:
        k = min(self.config.tournament_size, len(population))
        contestants = self._rng.sample(range(len(population)), k)
        winner = max(contestants, key=lambda i: scores[i])
        return population[winner]

    def _order_crossover(
        self, parent1: List[int], parent2: List[int]
    ) -> Tuple[List[int], List[int]]:
        n = len(parent1)
        a, b = sorted(self._rng.sample(range(n), 2))

        def _ox(p1: List[int], p2: List[int]) -> List[int]:
            child: List[Optional[int]] = [None] * n
            child[a : b + 1] = p1[a : b + 1]
            segment = set(p1[a : b + 1])
            fill_vals = [x for x in p2 if x not in segment]
            fill_idx = 0
            for i in range(n):
                if child[i] is None:
                    child[i] = fill_vals[fill_idx]
                    fill_idx += 1
            return child  # type: ignore[return-value]

        return _ox(parent1, parent2), _ox(parent2, parent1)

    def _mutate(self, chromosome: List[int]) -> List[int]:
        r = self._rng.random()
        if r < 0.25:
            return self._adjacent_swap_mutation(chromosome)
        elif r < 0.50:
            return self._short_inversion_mutation(chromosome)
        elif r < 0.75:
            return self._or_opt_1_mutation(chromosome)
        else:
            return self._swap_mutation(chromosome)

    def _adjacent_swap_mutation(self, chromosome: List[int]) -> List[int]:
        c = chromosome[:]
        if len(c) < 2:
            return c
        i = self._rng.randrange(len(c) - 1)
        c[i], c[i + 1] = c[i + 1], c[i]
        return c

    def _short_inversion_mutation(self, chromosome: List[int]) -> List[int]:
        c = chromosome[:]
        n = len(c)
        if n < 2:
            return c
        seg_len = self._rng.randint(2, min(6, n))
        i = self._rng.randrange(n - seg_len + 1)
        c[i : i + seg_len] = c[i : i + seg_len][::-1]
        return c

    def _or_opt_1_mutation(self, chromosome: List[int]) -> List[int]:
        c = chromosome[:]
        n = len(c)
        if n < 2:
            return c
        i = self._rng.randrange(n)
        gene = c.pop(i)
        j = self._rng.randrange(n)
        c.insert(j, gene)
        return c

    def _swap_mutation(self, chromosome: List[int]) -> List[int]:
        c = chromosome[:]
        i, j = self._rng.sample(range(len(c)), 2)
        c[i], c[j] = c[j], c[i]
        return c
