"""
Triplet-codon genome → LoRA adapter → LM (TinyLlama-1B, Pythia-70M, etc.)
"""

from __future__ import annotations
from copy import deepcopy
from typing import List

import torch
import peft
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE       = "mps" if torch.backends.mps.is_available() else "cpu"
#BASE_MODEL = "EleutherAI/Pythia-70M"
BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v0.5"       # ≲6 GB FP16 on M-series
GLOBAL_MAX_RANK = 512

_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
_base_master = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map=DEVICE,
    load_in_8bit=False,
).eval()


# ---------------------------------------------------------------------------
# Helper: pick target module names automatically
# ---------------------------------------------------------------------------
def _guess_target_modules(model) -> List[str]:
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    # GPT-NeoX / Pythia
    if any("query_key_value" in n for n in names):
        return ["query_key_value"]
    # Llama / Falcon
    if any("q_proj" in n for n in names):
        return ["q_proj", "v_proj"]
    # GPT-2
    if any("c_attn" in n for n in names):
        return ["c_attn"]
    # Fallback: first linear inside attention
    return [names[0]]


# ---------------------------------------------------------------------------
def genome_to_lora(genome: bytes) -> peft.PeftModel:
    START, STOP, CODON = 0xAAAAAA, 0xFFFFFF, 3
    i, started = 0, False
    rank_acc, dropout_p, alpha = 0, 0.1, 16
    adapters: List[dict] = []

    while i + CODON <= len(genome):
        codon = int.from_bytes(genome[i : i + CODON], "big")
        i += CODON

        if not started:
            started = codon == START
            continue
        if codon == STOP:
            break

        top    = (codon & 0xFF0000) >> 16
        middle = (codon & 0x00FF00) >> 8

        if   top == 0x01: rank_acc += max(1,  int(  4 * (1 + middle / 255)))
        elif top == 0x05: rank_acc += max(4,  int( 16 * (1 + middle / 255)))
        elif top == 0x08: rank_acc += max(8,  int( 32 * (1 + middle / 255)))

        elif top == 0x02 and rank_acc:      # commit
            adapters.append(
                {"rank": rank_acc, "lora_alpha": alpha, "lora_dropout": dropout_p}
            )
            rank_acc = 0
        elif top == 0x03 and adapters:      # duplicate
            adapters.append(adapters[-1].copy())
        elif top == 0x04: dropout_p = max(0.05, min(0.5, middle / 255))
        elif top == 0x06: alpha     = 8 + 24 * (middle / 255)

    if rank_acc:
        adapters.append(
            {"rank": rank_acc, "lora_alpha": alpha, "lora_dropout": dropout_p}
        )
    if not adapters:
        raise ValueError("empty adapter list")

    # cap rank
    hidden_dim  = _base_master.config.hidden_size
    max_rank    = min(GLOBAL_MAX_RANK, hidden_dim)
    adapters[0]["rank"] = max(4, min(adapters[0]["rank"], max_rank))

    # auto-detect target modules
    tgt = _guess_target_modules(_base_master)

    peft_cfg = peft.LoraConfig(
        r              = adapters[0]["rank"],
        lora_alpha     = adapters[0]["lora_alpha"],
        lora_dropout   = adapters[0]["lora_dropout"],
        bias           = "none",
        task_type      = peft.TaskType.CAUSAL_LM,
        target_modules = tgt,
    )

    model = peft.get_peft_model(deepcopy(_base_master), peft_cfg).to(DEVICE)
    return model
