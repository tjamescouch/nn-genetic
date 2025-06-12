import time
from triplet_encoding import random_genome;
from fitness_lora import fitness

t0 = time.time()

fitness(random_genome())   # one genome

print("seconds:", time.time() - t0)