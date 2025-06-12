import math
import random
from pathlib import Path

import datasets
import torch
from lora_encoding import DEVICE, _tokenizer, genome_to_lora

# ---------------------------------------------------------------------------
# Dataset cache ( ./data/hf_cache )
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR    = PROJECT_ROOT / "data" / "hf_cache"
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

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BAD_FITNESS   = -1e8    # returned when genome explodes / throws
STEPS         = 400     # micro-fine-tune iterations
LR            = 1e-3
CTX           = 512     # context length ( ≤ 1024 )
PARAM_PENALTY = 1e-5    # fitness -= coeff × √params


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------
def fitness(genome: bytes, steps: int = STEPS) -> float:
    """Return fitness for **one** genome.

    Procedure
    ---------
    1. Build LoRA-wrapped model from genome.
    2. Train *steps* on slice-1.
    3. Measure loss on **slice-1** *and* a fresh **slice-2**.
    4. Fitness = −(avg_loss) − penalty·√params  (higher is better).
    """

    # ---- build / guard -----------------------------------------------------
    try:
        model = genome_to_lora(genome).to(DEVICE)
    except Exception:
        return BAD_FITNESS

    # keep only LoRA weights trainable
    trainable_params = [
        p for n, p in model.named_parameters() if "lora" in n.lower()
    ]
    if not trainable_params:
        return BAD_FITNESS

    for p in model.parameters():
        p.requires_grad_(False)
    for p in trainable_params:
        p.requires_grad_(True)

    opt        = torch.optim.Adam(trainable_params, lr=LR)
    criterion  = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---- helper to grab a random CTX-token slice --------------------------
    def random_slice():
        idx = random.randint(0, TOKENS.size(1) - (CTX + 1))
        x   = TOKENS[:, idx : idx + CTX]
        y   = TOKENS[:, idx + 1 : idx + CTX + 1]
        return x, y

    # slice-1 for training
    x1, y1 = random_slice()

    # ---- micro fine-tune on slice-1 ---------------------------------------
    model.train()
    for _ in range(steps):
        out = model(x1)
        logits = out.logits.float()
        loss   = criterion(logits.view(-1, logits.size(-1)), y1.view(-1))
        if not torch.isfinite(loss):
            return BAD_FITNESS
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()

    # ---- evaluate on slice-1 and fresh slice-2 ----------------------------
    model.eval()
    with torch.no_grad():
        loss1 = loss.item()  # last training loss
        x2, y2 = random_slice()
        logits2 = model(x2).logits.float()
        loss2 = criterion(logits2.view(-1, logits2.size(-1)), y2.view(-1)).item()
        loss_val = 0.5 * (loss1 + loss2)

    # ---- size penalty (√params) ------------------------------------------
    trainable = sum(p.numel() for p in trainable_params)
    param_penalty = PARAM_PENALTY * math.sqrt(trainable)

    # debug – comment out for long GA runs
    print(f"debug loss1={loss1:.3f} loss2={loss2:.3f}  trainable={trainable}")

    return -loss_val - param_penalty
