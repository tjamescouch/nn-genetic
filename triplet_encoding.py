# triplet_encoding.py
"""
Triplet-codon genome  →  PyTorch phenotype
-----------------------------------------

• 1 codon = 3 bytes (24 b, MSB→LSB)  
• START = 0xAAAAAA  |  STOP = 0xFFFFFF  
• top-byte  = opcode  
• middle-byte = per-opcode parameter (0–255, clipped if unused)  
• low-byte = neutral drift / synonym padding

Opcode summary
──────────────
top  | action                                      | middle-byte meaning
─────|---------------------------------------------|-------------------------------
0x01 | add hidden units  +32 × (1 + m/255)         | fine-grained width step
0x05 | add hidden units  +128                      | synonyms m=0–255
0x08 | add hidden units  +256                      | synonyms m=0–255
0x02 | commit Dense → ReLU block                   | flush pending units
0x03 | duplicate previous Dense → Act (width-safe) | —
0x04 | Dropout(p = 0.05 + m/510) ∈ [0.05, 0.55]    | dropout rate
0x06 | BatchNorm1d  (mom = 0.05 + m/510)           | momentum
0x07 | Tanh activation (gain = 0.5 + m/255)        | optional α scaling
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

        elif top_byte == 0x04:   # dropout with encoded p
            p = max(0.05, min(0.5, middle / 255))
            layers.append(nn.Dropout(p))

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
