"""Train nano-dates from scratch — prompt-masked SFT on code-generated data.

    python train.py                 # ~30s on a GPU, a few minutes on CPU
    python train.py --steps 3000    # quicker, lower accuracy

Saves weights to model.safetensors (the tied lm_head is dropped and reconstructed
on load). No framework: just torch + numpy + safetensors and the two local files.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from generate_data import IGNORE_INDEX, date_pairs, encode_pair
from modeling_nano_dates import NanoDates

_VAL_SEED_OFFSET = 7919


def _encode_pool(pairs, seq_len):
    xs = torch.empty((len(pairs), seq_len), dtype=torch.long)
    ys = torch.empty((len(pairs), seq_len), dtype=torch.long)
    for i, (prompt, target) in enumerate(pairs):
        x, y = encode_pair(prompt, target, seq_len)
        xs[i] = torch.tensor(x); ys[i] = torch.tensor(y)
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", default="model.safetensors")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--n_train", type=int, default=100_000)
    ap.add_argument("--n_val", type=int, default=4_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    with open(args.config) as f:
        cfg = json.load(f)
    model = NanoDates(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    # Pre-encode the whole pool once (a nano model barely loads the GPU, so per-batch
    # Python tokenisation would dominate). Training is then a pure tensor gather.
    train_x, train_y = _encode_pool(date_pairs(args.seed, args.n_train), args.seq_len)
    val_x, val_y = _encode_pool(date_pairs(args.seed + _VAL_SEED_OFFSET, args.n_val), args.seq_len)
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    gen = torch.Generator(device=device).manual_seed(args.seed + 11)

    def lr_at(step):  # linear warmup -> cosine decay to 0
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * args.lr * (1 + math.cos(math.pi * p))

    model.train()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = torch.randint(0, train_x.shape[0], (args.batch_size,), generator=gen, device=device)
        logits = model(train_x[idx])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), train_y[idx].reshape(-1),
                               ignore_index=IGNORE_INDEX)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == args.steps - 1:
            model.eval()
            with torch.no_grad():
                vl = model(val_x[:2000])
                vloss = F.cross_entropy(vl.reshape(-1, vl.size(-1)), val_y[:2000].reshape(-1),
                                        ignore_index=IGNORE_INDEX).item()
            model.train()
            print(f"step {step:6d}  loss {loss.item():.4f}  val_loss {vloss:.4f}")

    # Drop the tied lm_head (shares storage with tok_emb; reconstructed on load).
    sd = {k: v.detach().cpu().contiguous().float()
          for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(sd, args.out, metadata={"format": "pt", "tied": "lm_head.weight=tok_emb.weight"})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
