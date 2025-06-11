# main.py
import random, math, copy, torch, torch.nn.functional as F
from triplet_encoding import (random_genome, mutate, genome_to_net,
                               INPUT_SIZE, OUTPUT_CLASSES, CODON_LEN)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------- hyperparams ----------------
POP_SIZE       = 30
GENERATIONS    = 10
MUT_P          = 0.03        # per-byte mutation prob
TOURN_SIZE     = 4           # tournament selection
BATCH_SIZE     = 128
EPOCHS         = 4           # per genome
CROSSOVER_RATE = 0.3
DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"

# ---------------- data ----------------
train_loader = DataLoader(
    datasets.MNIST("data", download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

def fitness(genome):
    try:
        net = genome_to_net(genome).to(DEVICE)
        x, y = next(iter(train_loader))
        logits = net(x.view(x.size(0), -1).to(DEVICE))   # dummy pass
    except Exception:
        return 0.0        # malformed phenotype → dead genome

    opt = torch.optim.SGD(net.parameters(), lr=0.1)

    correct = total = 0
    for _ in range(EPOCHS):
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(DEVICE)
            y = y.to(DEVICE)

            logits = net(x)
            loss   = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total  # accuracy 0-1

def crossover(parent_a: bytes, parent_b: bytes) -> bytes:
    """One-point crossover at a codon boundary (keeps START/STOP intact)."""
    # choose a cut AFTER the START codon and BEFORE the STOP codon in parent_a
    max_cut_codons = (len(parent_a) // CODON_LEN) - 2        # reserve START+STOP
    if max_cut_codons <= 1:
        return parent_a  # genome too short to cut

    cut_codon = random.randrange(1, max_cut_codons)  # 1 .. max-1
    cut_byte  = cut_codon * CODON_LEN
    return parent_a[:cut_byte] + parent_b[cut_byte:]

def reproduce(population, scores, mutation_p, crossover_rate=0.3):
    """Return a child genome using tournament + crossover + mutation."""
    if random.random() < crossover_rate:
        parent_a = tournament_select(population, scores)
        parent_b = tournament_select(population, scores)
        child = crossover(parent_a, parent_b)
    else:
        parent = tournament_select(population, scores)
        child  = parent

    # always mutate (can set p=0.0 if you want “pure” offspring)
    child = mutate(child, p=mutation_p)
    return child

def tournament_select(pop, scores):
    """Pick one genome via k-way tournament."""
    contenders = random.sample(list(range(len(pop))), TOURN_SIZE)
    best = max(contenders, key=lambda i: scores[i])
    return copy.deepcopy(pop[best])

# ---------------- GA loop ----------------
population = [random_genome() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    scores = [fitness(g) for g in population]
    best_i = max(range(POP_SIZE), key=lambda i: scores[i])
    valid = sum(s > 0 for s in scores)
    print(f"gen {gen:02d} best_acc {scores[best_i]:.3f} viable {valid}/{POP_SIZE}")
    best_net = genome_to_net(population[best_i])
    param_count = sum(p.numel() for p in best_net.parameters())
    print("  layers:", len(list(best_net)), "params:", param_count//1_000, "k")


    next_pop = []
    while len(next_pop) < POP_SIZE:
        child = reproduce(population, scores, MUT_P, crossover_rate=CROSSOVER_RATE)
        next_pop.append(child)
    population = next_pop

best_genome = max(population, key=fitness)
print("\n>>> BEST ARCHITECTURE")
print(genome_to_net(best_genome))
