# main.py – GA driver with rank‑curriculum and micro‑evolution
"""Run with:
    python main.py
Prints best fitness, viable genomes, and current rank cap per generation.
"""

import random
import copy

import lora_encoding            # GLOBAL_MAX_RANK will be updated each gen
import fitness_lora as fitmod   # we mutate its PARAM_PENALTY at runtime
from triplet_encoding import random_genome, mutate

# ---------------------------------------------------------------------------
# GA hyper‑parameters
# ---------------------------------------------------------------------------
POP_SIZE    = 30
GENERATIONS = 60
MUT_P       = 0.05
CROSS_RATE  = 0.5
CODON_BYTES = 3

# ---------------------------------------------------------------------------
# One‑point crossover helper
# ---------------------------------------------------------------------------

def crossover(a: bytes, b: bytes) -> bytes:
    max_cut = min(len(a), len(b)) // CODON_BYTES - 1
    if max_cut <= 1:
        return a  # genomes too short to cross
    cut = random.randrange(1, max_cut) * CODON_BYTES
    return a[:cut] + b[cut:]


# Tournament selection

def tournament(pop, scores, k: int = 4):
    cand = random.sample(range(len(pop)), k)
    return copy.deepcopy(pop[max(cand, key=lambda i: scores[i])])


def reproduce(pop, scores):
    child = (
        crossover(*random.sample(pop, 2))
        if random.random() < CROSS_RATE
        else tournament(pop, scores)
    )
    return mutate(child, p=MUT_P)

# ---------------------------------------------------------------------------
# Rank‑cap curriculum & penalty schedule
# (gen ≥ threshold) → (cap, penalty_coeff)
# ---------------------------------------------------------------------------
CURRICULUM = [
    (0,  128, 5e-6),
    (10, 256, 5e-6),
    (30, 512, 1e-5),
]


def adjust_curriculum(gen: int):
    """Update GLOBAL_MAX_RANK and PARAM_PENALTY based on current generation."""
    cap, coeff = CURRICULUM[0][1], CURRICULUM[0][2]
    for g, c, p in CURRICULUM:
        if gen >= g:
            cap, coeff = c, p
    lora_encoding.GLOBAL_MAX_RANK = cap
    fitmod.PARAM_PENALTY = coeff


# ---------------------------------------------------------------------------
# GA loop
# ---------------------------------------------------------------------------

def main():
    population = [random_genome() for _ in range(POP_SIZE)]

    for gen in range(GENERATIONS):
        adjust_curriculum(gen)

        scores = [fitmod.fitness(g) for g in population]
        valid  = sum(s > -1e8 for s in scores)            # BAD_FITNESS = -1e8
        best   = max(scores)

        print(
            f"gen {gen:02d}  best_fitness {best:.3f}  "
            f"viable {valid}/{POP_SIZE}  cap {lora_encoding.GLOBAL_MAX_RANK}"
        )

        # Reproduce next generation
        population = [reproduce(population, scores) for _ in range(POP_SIZE)]


if __name__ == "__main__":
    main()
