# Neural Genome Prototype

This repository contains a **minimal proof‑of‑concept** implementation of

* a compact, triplet‑codon genome (24‑bit codons with **START**/**STOP** markers),
* a deterministic decoder that maps genomes to PyTorch phenotypes, and
* a lightweight genetic algorithm with optional mutation & one‑point crossover.

The current phenotype target is a small MLP trained on MNIST.  The code is
*deliberately small* so we can validate the encoding and GA mechanics before
progressing to the real research goal: **evolving LoRA‑style adapters for
language models across low‑bandwidth peer‑to‑peer networks**.

---

## Quick start

```bash
# clone & set up
python -m venv venv && source venv/bin/activate
pip install torch torchvision
python main.py
```

The script prints the best accuracy and basic model stats each generation.  A
single run on an Apple M‑series CPU/GPU finishes in a couple of minutes.

---

## Directory layout

| Path                  | Purpose                       |
| --------------------- | ----------------------------- |
| `triplet_encoding.py` | Genome helpers & decoder      |
| `main.py`             | GA loop, fitness, logging     |
| `data/`               | MNIST cache (auto‑downloaded) |
| `README.md`           | You are here                  |

---

## Road‑map (incremental)

1. **Adapter phenotype** – swap the MLP decoder for a LoRA‑injection decoder
   targeting a 140 M‑parameter mini‑LM.
2. **Island model** – UDP gossip of elite genomes every *N* generations.
3. **Seed‑only genomes** – encode adapters via RNG seed + hyper‑parameters to
   shrink network traffic to a few bytes.
4. **Multi‑objective fitness** – trade off perplexity, adapter size, and
   inference latency.

---

## Prior art & inspiration

| Work                                                      | Relevance                                                           |
| --------------------------------------------------------- | ------------------------------------------------------------------- |
| **NEAT / HyperNEAT** (Stanley & Miikkulainen, 2002‑2009)  | Indirect encodings and evolving network topologies.                 |
| **Triplet‑Codon Encoding Method (TCENNE)** – *Genes* 2018 | Direct precedent for 24‑bit codon genomes and redundant codon sets. |
| **Grammatical Evolution** (O’Neill & Ryan, 2001)          | Demonstrated codon‑based genotype → grammar decoding.               |
| **LoRA (Hu et al., 2021)**                                | Rank‑limited adapter technique we plan to evolve.                   |

We claim no novelty in the individual components; the contribution we are
working toward is **combining a redundant triplet genome with LoRA adapter
search in a peer‑to‑peer evolutionary setting**.

---

## License

This work is released under the MIT License (see `LICENSE`).  Please check that
external datasets and pretrained models you plug in comply with their original
licenses.

---

## Citation

If you reference this prototype, please cite the relevant upstream papers above
and link back to this repository with the commit hash you used.
