# lora_encoding.py
"""
Triplet-codon genome → LoRA adapter → mini Llama-compatible LM
--------------------------------------------------------------

High-byte opcodes
─────────────────
0x01 / 0x05 / 0x08   add-rank   Δr = base × (1 + middle/255)
0x02                 commit adapter (inject into next Linear/Conv)
0x03                 duplicate previous adapter spec
0x04                 set dropout-p      p = 0.05 – 0.50
0x06                 set alpha-scale    α = 8 – 32
0x07                 reserved
"""

from __future__ import annotations
from typing import List
from copy import deepcopy
import torch, peft
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── env & base model ────────────────────────────────────────────────────────
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v0.5"       # ≲6 GB FP16 on M-series
GLOBAL_MAX_RANK = 512                                    # per-layer cap

_tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL)
_base_master = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map=DEVICE,
    load_in_8bit=False          # no bitsandbytes on macOS
).eval()                        # keep a frozen copy to deepcopy later

# ── genome → PEFT model ─────────────────────────────────────────────────────
def genome_to_lora(genome: bytes) -> peft.PeftModel:
    START, STOP, CODON = 0xAAAAAA, 0xFFFFFF, 3
    i, started = 0, False
    rank_acc, dropout_p, alpha = 0, 0.10, 16
    adapters: List[dict] = []

    while i + CODON <= len(genome):
        codon = int.from_bytes(genome[i : i + CODON], "big")
        i += CODON

        if not started:
            started = (codon == START)
            continue
        if codon == STOP:
            break

        top    = (codon & 0xFF0000) >> 16
        middle = (codon & 0x00FF00) >> 8

        # --- accumulate or set hyper-params ---------------------------------
        if   top == 0x01:  rank_acc += max(1, int( 4 * (1 + middle/255)))
        elif top == 0x05:  rank_acc += max(4, int(16 * (1 + middle/255)))
        elif top == 0x08:  rank_acc += max(8, int(32 * (1 + middle/255)))

        elif top == 0x02:                                 # commit adapter
            if rank_acc:
                adapters.append(
                    {"rank":        rank_acc,
                     "lora_alpha":  alpha,
                     "lora_dropout": dropout_p}
                )
                rank_acc = 0

        elif top == 0x03 and adapters:                    # duplicate spec
            adapters.append(adapters[-1].copy())

        elif top == 0x04:  dropout_p = max(0.05, min(0.5, middle/255))
        elif top == 0x06:  alpha     = 8 + 24 * (middle / 255)

    # commit trailing rank, if any
    #if rank_acc:
    #    adapters.append(
    #        {"rank":        rank_acc,
    #         "lora_alpha":  alpha,
    #         "lora_dropout": dropout_p}
    #    )

    if not adapters:
        raise ValueError("empty adapter list – genome produced no adapters")

    # ── prepare PEFT config (cap rank vs. hidden size) ──────────────────────
    hidden_dim = _base_master.config.hidden_size
    max_rank   = min(GLOBAL_MAX_RANK, hidden_dim)

    first = adapters[0]
    first["rank"]        = max(4, min(first["rank"], max_rank))
    first["lora_alpha"]  = max(1, first["lora_alpha"])

    peft_cfg = peft.LoraConfig(
        r              = first["rank"],
        lora_alpha     = first["lora_alpha"],
        lora_dropout   = first["lora_dropout"],
        bias           = "none",
        task_type      = peft.TaskType.CAUSAL_LM,
        target_modules = ["q_proj", "v_proj"],
    )

    cpu_base = _base_master.to("cpu")
    model = peft.get_peft_model(deepcopy(cpu_base), peft_cfg).to(DEVICE)

    for p in model.base_model.parameters():    # freeze backbone (safety)
        p.requires_grad_(False)

    model.train()        # enable gradients for new adapter params only
    return model