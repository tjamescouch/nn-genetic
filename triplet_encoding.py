# triplet_encoding.py
"""
Triplet-codon genome ⇨ PyTorch phenotype
=======================================

* Each codon = 3 bytes (24 bits) read MSB→LSB.
* START  = 0xAAAAAA   - begins decoding
* STOP   = 0xFFFFFF   - terminates decoding
* Top    = high-order byte  (first of three)
* Middle = second byte
* Low    = third byte  (neutral drift, ignored)

Opcode table
------------
Top byte   Meaning                     Synonyms (middle byte)       Effect
0x01       add +32 hidden units        0x00-0xFF                    accumulate
0x05       add +128 hidden units       0x00-0xFF                    accumulate
0x08       add +256 hidden units       0x00-0xFF                    layer
0x02       commit Dense→ReLU block     0x00-0xFF                    flush
0x03       duplicate previous block    0x00-0xFF                    copy
0x04       Dropout(p=0.10)             0x00-0x7F                    layer
0x14       Dropout(p=0.25)             0x80-0xFF                    layer
0x06       BatchNorm1d                 0x00-0xFF                    layer
0x07       Tanh activation             0x00-0xFF                    layer
"""
from __future__ import annotations
import random, torch, torch.nn as nn
from typing import List

# ---------- constants ----------
CODON_LEN      = 3
START_CODON    = 0xAAAAAA
STOP_CODON     = 0xFFFFFF
INPUT_SIZE     = 28 * 28     # MNIST
OUTPUT_CLASSES = 10
NEUTRAL_MASK   = 0xFF00FF    # high+low byte → opcode; middle byte = drift

# ---------- helpers ----------
def _bytes_to_codon(b: bytes, idx: int) -> int:
    return int.from_bytes(b[idx : idx + CODON_LEN], "big")

def random_genome(n_codons: int = 60) -> bytes:
    """Random genome with at least one valid Dense block."""
    body = [START_CODON,
            0x010000,               # +32
            0x020000]               # commit
    body.extend(random.randrange(0x000000, 0xFFFFFF) for _ in range(n_codons))
    body.append(STOP_CODON)
    return b"".join(c.to_bytes(3, "big") for c in body)

def mutate(genome: bytes, p: float = 0.02) -> bytes:
    """Byte-wise point mutation."""
    g = bytearray(genome)
    for i in range(len(g)):
        if random.random() < p:
            g[i] = random.randrange(256)
    return bytes(g)

# ---------- decoding ----------
def genome_to_net(genome: bytes) -> nn.Module:
    """Decode a triplet-codon genome into a torch.nn.Sequential network."""
    def last_linear_out() -> int:
        """Return out_features of the most-recent Linear (or INPUT_SIZE if none)."""
        for layer in reversed(layers):
            if isinstance(layer, nn.Linear):
                return layer.out_features
        return INPUT_SIZE

    i, started, pending_h = 0, False, 0
    layers: List[nn.Module] = []

    while i + CODON_LEN <= len(genome):
        codon = _bytes_to_codon(genome, i)
        i += CODON_LEN

        if not started:
            if codon == START_CODON:
                started = True
            continue
        if codon == STOP_CODON:
            break

        top_byte = (codon & 0xFF0000) >> 16

        # -------- opcode actions --------
        if   top_byte == 0x01:            # +32
            pending_h += 32

        elif top_byte == 0x05:            # +128
            pending_h += 128

        elif top_byte == 0x08:            # +256
            pending_h += 256

        elif top_byte == 0x02:            # commit Dense→ReLU
            if pending_h:
                in_f = last_linear_out()
                layers += [nn.Linear(in_f, pending_h), nn.ReLU()]
                pending_h = 0

        elif top_byte == 0x03:            # duplicate prev Dense→ReLU
            # find last Linear layer
            for idx in range(len(layers) - 1, -1, -1):
                if isinstance(layers[idx], nn.Linear):
                    lin = layers[idx]
                    layers += [nn.Linear(lin.in_features, lin.out_features),
                               nn.ReLU()]
                    break

        elif top_byte == 0x04:            # Dropout 0.10
            layers.append(nn.Dropout(0.10))

        elif top_byte == 0x14:            # Dropout 0.25
            layers.append(nn.Dropout(0.25))

        elif top_byte == 0x06:            # BatchNorm1d
            layers.append(nn.BatchNorm1d(last_linear_out()))

        elif top_byte == 0x07:            # Tanh
            layers.append(nn.Tanh())

    # flush any remaining hidden units
    if pending_h:
        layers += [nn.Linear(last_linear_out(), pending_h), nn.ReLU()]

    # final classifier
    layers.append(nn.Linear(last_linear_out(), OUTPUT_CLASSES))
    return nn.Sequential(*layers)


# ---------- tiny smoke test ----------
if __name__ == "__main__":
    torch.manual_seed(0)
    g = random_genome()
    net = genome_to_net(g)
    print(net)
