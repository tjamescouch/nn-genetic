"""
lora_encoding.py
────────────────
Triplet-codon genome → LoRA adapter → Hugging-Face causal-LM.

• Supports TinyLlama-1B, Pythia-70M, Falcon-1B, GPT-2, etc.
• Target linear modules auto-detected.
• `GLOBAL_MAX_RANK` can be raised at runtime by main.py’s curriculum.
"""

from __future__ import annotations
from copy import deepcopy
from typing import List

import torch
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Base model (small & fast by default; override here if desired)
# ---------------------------------------------------------------------------
#BASE_MODEL = "EleutherAI/Pythia-70M"
BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v0.5"       # ≲6 GB FP16 on M-series
DEVICE            = "mps" if torch.backends.mps.is_available() else "cpu"
GLOBAL_MAX_RANK   = 128                         # will be modified by main.py

_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
_base_master = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map=DEVICE,
    load_in_8bit=False,
).eval()


# ---------------------------------------------------------------------------
# Helper: guess which Linear modules to LoRA-inject
# ---------------------------------------------------------------------------
def _guess_target_modules(model) -> List[str]:
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    if any("query_key_value" in n for n in names):      # GPT-NeoX / Pythia
        return ["query_key_value"]
    if any("q_proj" in n for n in names):               # Llama / Falcon
        return ["q_proj", "v_proj"]
    if any("c_attn" in n for n in names):               # GPT-2
        return ["c_attn"]
    return [names[0]]                                   # fallback


# ---------------------------------------------------------------------------
def genome_to_lora(genome: bytes) -> peft.PeftModel:
    START, STOP, CODON = 0xAAAAAA, 0xFFFFFF, 3
    i, started = 0, False
    rank_acc, dropout_p, alpha = 0, 0.10, 16
    adapters: List[dict] = []

    while i + CODON <= len(genome):
        codon = int.from_bytes(genome[i : i + CODON], "big"); i += CODON

        if not started:
            started = codon == START
            continue
        if codon == STOP:
            break

        top    = (codon & 0xFF0000) >> 16
        middle = (codon & 0x00FF00) >> 8

        if   top == 0x01: rank_acc += max(1,  int(  4 * (1 + middle/255)))
        elif top == 0x05: rank_acc += max(4,  int( 16 * (1 + middle/255)))
        elif top == 0x08: rank_acc += max(8,  int( 32 * (1 + middle/255)))

        elif top == 0x02 and rank_acc:                     # commit adapter
            adapters.append(
                {"rank": rank_acc, "lora_alpha": alpha, "lora_dropout": dropout_p}
            )
            rank_acc = 0
        elif top == 0x03 and adapters:                     # duplicate
            adapters.append(adapters[-1].copy())
        elif top == 0x04: dropout_p = max(0.05, min(0.5, middle / 255))
        elif top == 0x06: alpha     = 8 + 24 * (middle / 255)

    if rank_acc:
        adapters.append(
            {"rank": rank_acc, "lora_alpha": alpha, "lora_dropout": dropout_p}
        )
    if not adapters:
        raise ValueError("empty adapter list")

    # -----------------------------------------------------------------------
    # Cap rank using current GLOBAL_MAX_RANK (can change each generation)
    # -----------------------------------------------------------------------
    hidden_dim = _base_master.config.hidden_size
    max_rank   = min(GLOBAL_MAX_RANK, hidden_dim)
    for ad in adapters:
        ad["rank"] = max(4, min(ad["rank"], max_rank))

    tgt_modules = _guess_target_modules(_base_master)

    cfg = peft.LoraConfig(
        r              = adapters[0]["rank"],
        lora_alpha     = adapters[0]["lora_alpha"],
        lora_dropout   = adapters[0]["lora_dropout"],
        bias           = "none",
        task_type      = peft.TaskType.CAUSAL_LM,
        target_modules = tgt_modules,
    )

    model = peft.get_peft_model(deepcopy(_base_master), cfg).to(DEVICE)
    return model
