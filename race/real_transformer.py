"""Two checks on real transformers with vector weights. Needs PyTorch. All gradients are exact (full batch).

  scale   A one-layer transformer on aligned RRXOR (width 32, 2 heads, 64 ReLU neurons, learned position
          embeddings, no layer norm). Every weight matrix starts as a Gaussian of standard deviation 0.1 * s.
          The first circuit's path crosses six matrices that start small: token embedding, V, O, MLP in,
          MLP out, unembedding. (The attention pattern starts uniform, and the MLP's direct read of the
          current token enters only through a sign.) Six factors predict a wait going as s^-4 under plain
          gradient descent, and a wait nearly independent of s under Adam.
  phase   A four-layer transformer on RRXOR started at a random phase (width 64, 4 heads, layer norm).
          Reference losses in bits per token: guessing 1.000, XOR only 0.926, XOR plus the phase bundle
          0.880, optimum 0.8234. Reports the step at which the loss passes 0.97, 0.91, 0.88 and 0.84
          under Adam and under plain SGD.

  rate    The one-layer transformer of "scale" under Adam at five learning rates. If each weight moved one
          learning rate per step, the step of the first drop times the rate would be constant.

Run:  python real_transformer.py scale          (about three minutes; add "small" for s = 0.25 and 0.18, about 25 more)
      python real_transformer.py rate           (about two minutes)
      python real_transformer.py phase          (Adam: about five minutes. Add "sgd" for the plain-SGD runs: hours)
"""
import itertools
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

L = 12


# ---------------------------------------------------------------- scale: one layer, aligned RRXOR

D1, H1, M1 = 32, 2, 64


def aligned():
    free = [p for p in range(1, L + 1) if p % 3]
    seqs = []
    for bits in itertools.product((0, 1), repeat=len(free)):
        s, it = [], iter(bits)
        for p in range(1, L + 1):
            s.append(s[-1] ^ s[-2] if p % 3 == 0 else next(it))
        seqs.append(s)
    return torch.tensor(seqs)


def make(seed, std):
    torch.manual_seed(seed)
    shapes = dict(tok=(2, D1), pos=(L, D1), Q=(D1, D1), K=(D1, D1), V=(D1, D1), O=(D1, D1), win=(M1, D1), wout=(D1, M1), U=(2, D1))
    P = {k: (torch.randn(*shape) * std).requires_grad_() for k, shape in shapes.items()}
    P['bin'] = torch.zeros(M1, requires_grad=True)
    return P


def loss_one_layer(P, X, mask):
    x = X[:, :-1]
    B, T = x.shape
    r = P['tok'][x] + P['pos'][:T]
    q = (r @ P['Q'].t()).view(B, T, H1, -1).transpose(1, 2)
    k = (r @ P['K'].t()).view(B, T, H1, -1).transpose(1, 2)
    v = (r @ P['V'].t()).view(B, T, H1, -1).transpose(1, 2)
    att = (q @ k.transpose(-1, -2) / math.sqrt(D1 // H1)).masked_fill(mask, float('-inf')).softmax(-1)
    r = r + (att @ v).transpose(1, 2).reshape(B, T, D1) @ P['O'].t()
    a = F.relu(r @ P['win'].t() + P['bin'])
    logits = (r + a @ P['wout'].t()) @ P['U'].t()
    return F.cross_entropy(logits.reshape(-1, 2), X[:, 1:].reshape(-1)) / math.log(2)


def first_drop(X, mask, seed, std, kind, lr, max_steps):
    """Step at which the loss first falls under 0.95 bits per token."""
    P = make(seed, std)
    opt = torch.optim.Adam(P.values(), lr) if kind == 'adam' else torch.optim.SGD(P.values(), lr)
    for step in range(max_steps):
        opt.zero_grad()
        ls = loss_one_layer(P, X, mask)
        if ls.item() < 0.95:
            return step
        ls.backward()
        opt.step()
    return None


def part_scale(small=False):
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    X = aligned()
    mask = torch.triu(torch.ones(L - 1, L - 1, dtype=torch.bool), 1)
    print("=== scale: one-layer transformer, aligned RRXOR, exact gradients; every weight matrix starts at std 0.1 * s ===")
    print("    step at which the loss first falls under 0.95 bits; plain gradient descent with learning rate 1, Adam with 0.001")
    print(f"{'s':>6s}   {'gradient descent, seeds':<28s} {'Adam, seeds 0-2':<18s}")
    plan = [(1.4, 3, 5000), (1.0, 3, 5000), (0.7, 3, 20000), (0.5, 3, 60000), (0.35, 3, 200000)]
    if small:
        plan += [(0.25, 3, 600000), (0.18, 1, 1500000)]
    rows = []
    for s, seeds, cap in plan:
        gd = [first_drop(X, mask, seed, 0.1 * s, 'sgd', 1.0, cap) for seed in range(seeds)]
        ad = [first_drop(X, mask, seed, 0.1 * s, 'adam', 1e-3, 20000) for seed in range(3)]
        rows.append((s, gd, ad))
        print(f"{s:6.2f}   {str(gd):<28s} {str(ad):<18s}", flush=True)
    print("    local exponent of the wait in 1/s, seed 0:")
    for (s1, g1, a1), (s2, g2, a2) in zip(rows, rows[1:]):
        print(f"      s = {s1} to {s2}:  gradient descent {math.log(g2[0] / g1[0]) / math.log(s1 / s2):.2f}   Adam {math.log(a2[0] / a1[0]) / math.log(s1 / s2):.2f}")
    torch.set_default_dtype(torch.float32)


def part_rate():
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    X = aligned()
    mask = torch.triu(torch.ones(L - 1, L - 1, dtype=torch.bool), 1)
    print("=== rate: one-layer transformer, aligned RRXOR, Adam with exact gradients, at five learning rates ===")
    print("    step at which the loss first falls under 0.95, seeds 0-2")
    for std in (0.05, 0.025):
        for lr in (2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4):
            r = [first_drop(X, mask, seed, std, 'adam', lr, 20000) for seed in range(3)]
            print(f"  initial std {std}, learning rate {lr}: {r}   step x rate: " + ', '.join(f"{a * lr:.3f}" for a in r), flush=True)
    torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------- phase: four layers, random-start RRXOR

def random_start():
    seqs, ws = [], []
    for o in (0, 1, 2):
        n = L + o
        free = [p for p in range(1, n + 1) if p % 3]
        for bits in itertools.product((0, 1), repeat=len(free)):
            s, it = [], iter(bits)
            for p in range(1, n + 1):
                s.append(s[-1] ^ s[-2] if p % 3 == 0 else next(it))
            seqs.append(s[o:])
            ws.append(1.0 / (3 * 2 ** len(free)))
    return torch.tensor(seqs), torch.tensor(ws)


class Block(nn.Module):
    def __init__(self, d, h, norm=True):
        super().__init__()
        self.h = h
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.n1, self.n2 = (nn.LayerNorm(d), nn.LayerNorm(d)) if norm else (nn.Identity(), nn.Identity())

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.out(F.relu(self.fc(self.n2(x))))


class Transformer(nn.Module):
    def __init__(self, d=64, h=4, layers=4, std=0.02, norm=True):
        """norm=False removes every layer norm; no part of the note uses that yet."""
        super().__init__()
        self.tok, self.pos = nn.Embedding(2, d), nn.Embedding(L, d)
        self.blocks = nn.ModuleList(Block(d, h, norm) for _ in range(layers))
        self.nf, self.un = (nn.LayerNorm(d) if norm else nn.Identity()), nn.Linear(d, 2)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0, std)
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        r = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        for b in self.blocks:
            r = b(r)
        return self.un(self.nf(r))


LEVELS = (0.97, 0.91, 0.88, 0.84)


def train_phase(X, W, seed, kind, lr, steps, std=0.02):
    """Steps at which the exact loss first passes each level, and the lowest loss reached."""
    torch.manual_seed(seed)
    net = Transformer(std=std)
    opt = torch.optim.Adam(net.parameters(), lr) if kind == 'adam' else torch.optim.SGD(net.parameters(), lr)
    hit, best = {}, 9.0
    for step in range(steps + 1):
        opt.zero_grad()
        ce = F.cross_entropy(net(X[:, :-1]).reshape(-1, 2), X[:, 1:].reshape(-1), reduction='none').view(len(X), -1).mean(1)
        ls = (W * ce).sum() / math.log(2)
        v = ls.item()
        if not math.isfinite(v):
            hit['diverged'] = step
            break
        best = min(best, v)
        for lv in LEVELS:
            if lv not in hit and v < lv:
                hit[lv] = step
        if LEVELS[-1] in hit:
            break
        ls.backward()
        opt.step()
    return hit, best


def part_phase(sgd=False):
    torch.set_num_threads(2)
    X, W = random_start()
    print("=== phase: four-layer transformer (width 64, 4 heads, layer norm), random-start RRXOR, exact gradients ===")
    print("    step at which the loss first passes 0.97 (XOR appearing), 0.91 (under the XOR-only level 0.926),")
    print("    0.88 (under XOR plus the phase bundle) and 0.84 (near the optimum 0.8234)")
    for seed in range(3):
        hit, best = train_phase(X, W, seed, 'adam', 1e-3, 3000)
        print(f"  Adam, learning rate 0.001, initial std 0.02, seed {seed}: {hit}   lowest loss {best:.4f}", flush=True)
    if sgd:
        for std, lr, steps in ((0.02, 0.3, 6000), (0.02, 1.0, 6000), (0.1, 0.1, 8000), (0.1, 0.3, 8000), (0.2, 0.1, 8000), (0.2, 0.3, 8000)):
            hit, best = train_phase(X, W, 0, 'sgd', lr, steps, std=std)
            print(f"  plain SGD, learning rate {lr}, initial std {std}, seed 0, {steps} steps: {hit}   lowest loss {best:.4f}", flush=True)
            if std != 0.02 and lr == 0.3:
                hit, best = train_phase(X, W, 0, 'adam', 1e-3, 3000, std=std)
                print(f"  Adam, learning rate 0.001, initial std {std}, seed 0: {hit}   lowest loss {best:.4f}", flush=True)


if __name__ == '__main__':
    args = sys.argv[1:] or ['scale']
    t0 = time.time()
    if args[0] == 'scale':
        part_scale(small='small' in args)
    elif args[0] == 'rate':
        part_rate()
    elif args[0] == 'phase':
        part_phase(sgd='sgd' in args)
    print(f"    ({time.time() - t0:.0f} s)")
