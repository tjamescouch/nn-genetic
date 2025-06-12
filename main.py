# main.py  – GA driver for LoRA evolution
import random, copy
from fitness_lora import fitness, BAD_FITNESS
from triplet_encoding import random_genome, mutate, crossover

POP_SIZE      = 30
GENERATIONS   = 20
MUT_P         = 0.05
CROSS_RATE    = 0.5
CODON_BYTES   = 3          # keep genomes cut-aligned

# ── GA helpers ──────────────────────────────────────────────────────────────
def tournament_select(pop, scores, k=4):
    cand = random.sample(range(len(pop)), k)
    return copy.deepcopy(pop[max(cand, key=lambda i: scores[i])])

def reproduce(pop, scores):
    if random.random() < CROSS_RATE:
        mum, dad = random.sample(pop, 2)
        child    = crossover(mum, dad)
    else:
        child    = tournament_select(pop, scores)
    return mutate(child, p=MUT_P)

# ── GA loop ─────────────────────────────────────────────────────────────────
population = [random_genome() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    scores = [fitness(g) for g in population]
    valid       = sum(s > BAD_FITNESS for s in scores)

    best_i = max(range(POP_SIZE), key=lambda i: scores[i])
    best   = scores[best_i]
    print(f"gen {gen:02d}  best_fitness {best:.3f}  viable {valid}/{POP_SIZE}")

    if all(s <= BAD_FITNESS for s in scores):      # every score is the penalty
        population = [random_genome() for _ in range(POP_SIZE)]
        continue

    population = [reproduce(population, scores) for _ in range(POP_SIZE)]