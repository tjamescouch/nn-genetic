import math
import random
from pathlib import Path

import datasets
import torch
from lora_encoding import DEVICE, _tokenizer, genome_to_lora

# ----------------------------------------------------------------------------
# Dataset cache (./data/hf_cache)
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_ROOT / "data" / "hf_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA = datasets.load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="train[:1%]",
    cache_dir=str(CACHE_DIR),
)
TOKENS = _tokenizer("\n\n".join(DATA["text"]), return_tensors="pt")[
    "input_ids"
].to(DEVICE)

# ----------------------------------------------------------------------------
# Hyper‑parameters
# ----------------------------------------------------------------------------
BAD_FITNESS = -1e8          # score for genomes that raise
STEPS = 400
LR    = 1e-3
PARAM_PENALTY = 1e-5        # gentler size penalty        # fitness -= coeff × trainable_params
CTX = 512                   # context length (≤1024)


# ----------------------------------------------------------------------------
# Fitness
# ----------------------------------------------------------------------------

def fitness(genome: bytes, steps: int = STEPS) -> float:
    """Return fitness for *one* genome.

    Score = -loss − penalty·params; higher (less‑negative) is better.
    """

    try:
        model = genome_to_lora(genome).to(DEVICE)
    except Exception:
        return BAD_FITNESS

    # freeze backbone; keep only LoRA params trainable
    trainable_params = []
    for name, p in model.named_parameters():
        if "lora" in name.lower():
            p.requires_grad_(True)
            trainable_params.append(p)
        else:
            p.requires_grad_(False)

    if not trainable_params:
        return BAD_FITNESS

    opt = torch.optim.Adam(trainable_params, lr=LR)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        # ---- sample TWO random CTX slices ----------------------------------
    def random_slice():
        idx = random.randint(0, TOKENS.size(1) - (CTX + 1))
        x   = TOKENS[:, idx : idx + CTX]
        y   = TOKENS[:, idx + 1 : idx + CTX + 1]
        return x, y

    # slice‑1 for training
    x1, y1 = random_slice()

    # ---- micro fine‑tune on slice‑1 -------------------------------------
    model.train()
    for _ in range(steps):
        out  = model(x1)
        logits = out.logits.float()
        loss1 = criterion(logits.view(-1, logits.size(-1)), y1.view(-1))
        if not torch.isfinite(loss1):
            return BAD_FITNESS
        opt.zero_grad(); loss1.backward();
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()

    # evaluate on slice‑1 (post‑train) and slice‑2 (fresh)
    model.eval()
    with torch.no_grad():
        loss1_val = loss1.item()
        x2, y2 = random_slice()
        logits2 = model(x2).logits.float()
        loss2_val = criterion(logits2.view(-1, logits2.size(-1)), y2.view(-1)).item()
        loss_val = 0.5 * (loss1_val + loss2_val)

    # ---- size penalty ----------------------------------------------------
    trainable = sum(p.numel() for p in trainable_params)
    param_penalty = PARAM_PENALTY * math.sqrt(trainable)

    print(f"debug loss1={loss1_val:.3f} loss2={loss2_val:.3f} trainable={trainable}")
    return -loss_val - param_penalty
