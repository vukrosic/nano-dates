"""Held-out exact-match eval, bucketed by phrase category.

    python eval.py                       # uses model.safetensors + config.json
    python eval.py --n 2000

Reports overall accuracy and a per-category breakdown, so you can see *which*
capabilities the model has — format-normalisation (absolute forms) vs calendar
arithmetic (relative forms). Decode is batched and greedy; it is bit-identical to
looping a single-prompt greedy decode, just far faster.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import torch

from generate_data import date_pairs
from modeling_nano_dates import NanoDates, load

_HOLDOUT_SEED = 987_654_321
_ISO_LEN = 10


def category(phrase: str) -> str:
    if re.match(r"\d{4}-\d{2}-\d{2}", phrase):
        return "abs:iso"
    if " of " in phrase:
        return "abs:ordinal"
    if re.match(r"[A-Z][a-z]+ \d", phrase):
        return "abs:long/abbr"
    if re.match(r"\d+ [A-Z]", phrase):
        return "abs:d-month-y"
    for w in ("today", "tomorrow", "yesterday", "next week", "last week", "next month"):
        if phrase == w:
            return f"rel:{w}"
    if "months" in phrase:
        return "rel:in-months"
    if "weeks" in phrase:
        return "rel:in-weeks"
    if "days ago" in phrase:
        return "rel:days-ago"
    if phrase.startswith("in") and "days" in phrase:
        return "rel:in-days"
    if phrase.startswith("next"):
        return "rel:next-weekday"
    if phrase.startswith("last"):
        return "rel:last-weekday"
    return "other"


@torch.no_grad()
def greedy_batch(model, prompts, n_new, device="cpu"):
    """Greedy-decode n_new tokens per prompt; bucket by length for a padding-free batch."""
    enc = [list(p.encode("utf-8")) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(enc[i]))
    max_seq = model.cfg["max_seq_len"]
    preds = [None] * len(prompts)
    i = 0
    while i < len(order):
        L = len(enc[order[i]])
        j = i
        while j < len(order) and len(enc[order[j]]) == L:
            j += 1
        rows = order[i:j]
        toks = torch.tensor([enc[k] for k in rows], dtype=torch.long, device=device)
        for _ in range(n_new):
            nxt = model(toks[:, -max_seq:])[:, -1, :].argmax(-1, keepdim=True)
            toks = torch.cat([toks, nxt], dim=1)
        tail = toks[:, L:L + n_new]
        for r, k in enumerate(rows):
            preds[k] = bytes(int(b) & 0xFF for b in tail[r].tolist()).decode("utf-8", "replace")
        i = j
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="model.safetensors")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load(args.weights, args.config, device=device)

    pairs = date_pairs(_HOLDOUT_SEED, args.n)
    preds = greedy_batch(model, [p for p, _ in pairs], _ISO_LEN, device=device)

    tot, ok = defaultdict(int), defaultdict(int)
    for (prompt, target), pred in zip(pairs, preds):
        phrase = prompt.split(" | ", 1)[1].rsplit(" => ", 1)[0]
        c = category(phrase)
        tot[c] += 1
        ok[c] += pred == target

    total_ok, total_n = sum(ok.values()), sum(tot.values())
    params = sum(p.numel() for p in model.parameters())
    print(f"params  : {params:,}")
    print(f"overall : {total_ok}/{total_n} = {total_ok/total_n:.1%}\n")
    print(f"{'category':<20}{'acc':>7}{'n':>6}")
    for c in sorted(tot, key=lambda k: ok[k] / tot[k]):
        print(f"{c:<20}{ok[c]/tot[c]:>6.0%}{tot[c]:>6}")


if __name__ == "__main__":
    main()
