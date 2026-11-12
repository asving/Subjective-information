"""Read the phase circuit out of a real transformer while it trains. Needs PyTorch and numpy.

The network is the four-layer transformer of real_transformer.py on random-start RRXOR, with exact
gradients. Its logit (class 1 minus class 0) is a function of the prefix it has seen. Every few steps
that function is fitted, by least squares weighted with the probability of each prefix, on

    a constant, the current token, and the six shapes raced in unit_race.py phase:
    xor        e_t = x_t * x_(t-1)
    two back   x_(t-2), a cheap trick
    cancel r   e_t * D_r for r = 2, 0, 1, with D_r = 1 when a token r, r+3, ... positions back broke the XOR rule
    sure       e_t * D_0 * D_1

The fitted coefficient of a shape is that unit's strength in the network, in the same units as the
strengths of the race. "explained" is the share of the logit's weighted variance that the fit accounts for;
what is left over is whatever the network computes that the six shapes do not.

Run:  python readout.py          (Adam, three seeds; about five minutes)
      python readout.py sgd      (plain SGD from weights five times larger, one seed; about seven minutes)
      python readout.py extra    (Adam again: what do the six shapes miss in the middle of training? Two larger
                                  fits are compared: the six plus e times a single violation at each offset 0 to 10
                                  back, and the six plus e times an indicator of each position. A third is a lookup
                                  table on e, the token two back, the three flags and the position mod 3, which
                                  contains every function of the columns the circuit uses)
      python readout.py sweep    (Adam at other learning rates and other values of beta2: when does each unit
                                  arrive? About twenty minutes)
"""
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import real_transformer as rt
import rrxor_random as rr

NAMES = ['xor', 'two back', 'cancel 2', 'cancel 0', 'cancel 1', 'sure']


def design():
    """Columns of the fit, one row per prefix, and the matrix that maps a logit vector to coefficients."""
    X = rr.X
    XOR = ('head', ('off', 1), X, X, 'mean')
    VIO = ('head', ('off', 1), X, XOR, 'mean')
    e = np.array(rr.val(XOR), dtype=float)
    D = {r: np.array([1.0 if m == 1 else 0.0 for m in rr.val(('head', ('per', 3, r), None, VIO, 'max'))]) for r in (2, 0, 1)}
    cols = [np.ones(len(rr.P)), np.array(rr.val(X), dtype=float), e,
            np.array(rr.val(('head', ('off', 2), None, X, 'mean')), dtype=float),
            e * D[2], e * D[0], e * D[1], e * D[0] * D[1]]
    Phi = np.stack(cols, axis=1)
    w = np.array(rr.W)
    M = np.linalg.solve(Phi.T @ (w[:, None] * Phi), Phi.T * w[None, :])
    return Phi, w, M


def extra_designs():
    """Two larger fits: the six shapes plus single violations, and the six shapes plus positions."""
    Phi, w, _ = design()
    X = rr.X
    XOR = ('head', ('off', 1), X, X, 'mean')
    VIO = ('head', ('off', 1), X, XOR, 'mean')
    e = np.array(rr.val(XOR), dtype=float)
    single = [e * np.array([1.0 if m == 1 else 0.0 for m in rr.val(('head', ('off', d), None, VIO, 'mean'))]) for d in range(11)]
    pos = [e * np.array([1.0 if len(p) == k else 0.0 for p in rr.P]) for k in range(2, rt.L)]
    out = []
    for cols in (single, pos):
        A = np.concatenate([Phi, np.stack(cols, axis=1)], axis=1)
        out.append((A, np.linalg.pinv(A.T @ (w[:, None] * A)) @ (A.T * w[None, :])))
    flags = [rr.val(('head', ('per', 3, r), None, VIO, 'max')) for r in (2, 0, 1)]
    back2 = rr.val(('head', ('off', 2), None, X, 'mean'))
    keys = {}
    group = np.array([keys.setdefault((e[i], back2[i], flags[0][i], flags[1][i], flags[2][i], len(p) % 3), len(keys)) for i, p in enumerate(rr.P)])
    return out, group


def representatives(Xs):
    """For every prefix of rrxor_random, one (sequence, position) of the training set that shows it."""
    rep = {}
    for i, row in enumerate(Xs.tolist()):
        seq = tuple(2 * b - 1 for b in row)
        for t in range(1, rt.L):
            rep.setdefault(seq[:t], (i, t - 1))
    idx = [rep[p] for p in rr.P]
    return torch.tensor([a for a, _ in idx]), torch.tensor([b for _, b in idx])


def train(kind, lr, std, seed, max_steps, every, stop=0.826, extra=None, beta2=0.999):
    """Train and read out. Returns rows (step, loss, explained, strengths of the six units[, shares of the larger fits])."""
    Xs, Ws = rt.random_start()
    Phi, w, M = design()
    ri, rp = representatives(Xs)
    torch.manual_seed(seed)
    net = rt.Transformer(std=std)
    opt = torch.optim.Adam(net.parameters(), lr, betas=(0.9, beta2)) if kind == 'adam' else torch.optim.SGD(net.parameters(), lr)
    rows = []
    for step in range(max_steps + 1):
        opt.zero_grad()
        logits = net(Xs[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, 2), Xs[:, 1:].reshape(-1), reduction='none').view(len(Xs), -1).mean(1)
        ls = (Ws * ce).sum() / math.log(2)
        v = ls.item()
        if not math.isfinite(v):
            break
        if step % every == 0 or v < stop:
            z = (logits[..., 1] - logits[..., 0])[ri, rp].detach().double().numpy()
            coef = M @ z
            res = z - Phi @ coef
            zbar = (w * z).sum() / w.sum()
            tot = (w * (z - zbar) ** 2).sum()
            row = (step, v, 1 - (w * res ** 2).sum() / tot if tot > 1e-12 else 0.0, coef[2:].copy())
            if extra:
                fits, group = extra
                row += tuple(1 - (w * (z - A @ (MA @ z)) ** 2).sum() / tot if tot > 1e-12 else 0.0 for A, MA in fits)
                mean = np.bincount(group, w * z) / np.bincount(group, w)
                row += (1 - (w * (z - mean[group]) ** 2).sum() / tot if tot > 1e-12 else 0.0,)
            rows.append(row)
        if v < stop:
            break
        ls.backward()
        opt.step()
    return rows


def report(title, rows, show=18):
    final = rows[-1][3]
    print(f"\n  {title}: ends at step {rows[-1][0]} with loss {rows[-1][1]:.4f}; the fit explains {100 * rows[-1][2]:.1f}% of the logit's variance")
    print(f"  {'step':>6s} {'loss':>7s} {'explained':>9s}  " + ' '.join(f"{n:>9s}" for n in NAMES))
    for row in rows[::max(1, len(rows) // show)] + rows[-1:]:
        print(f"  {row[0]:6d} {row[1]:7.4f} {100 * row[2]:8.1f}%  " + ' '.join(f"{x:+9.3f}" for x in row[3]))
    parts = []
    for j, n in enumerate(NAMES):
        pk = max(rows, key=lambda r: abs(r[3][j]))
        if abs(final[j]) >= 0.05:
            half = next(r[0] for r in rows if r[3][j] * final[j] > 0 and abs(r[3][j]) >= 0.5 * abs(final[j]))
            parts.append(f"{n}: half of final at step {half}, largest {pk[3][j]:+.2f} at {pk[0]}, final {final[j]:+.2f}")
        else:
            parts.append(f"{n}: largest {pk[3][j]:+.2f} at {pk[0]}, final {final[j]:+.2f}")
    print("    " + "\n    ".join(parts))
    late = [r[2] for r in rows if r[1] < 0.90]
    if late:
        print(f"    share explained while the loss is under 0.90: lowest {100 * min(late):.1f}%")
    sys.stdout.flush()


if __name__ == '__main__':
    torch.set_num_threads(4)
    t0 = time.time()
    print("=== readout: the six shapes of the phase circuit, fitted to a real four-layer transformer during training ===")
    if sys.argv[1:] == ['sweep']:
        print("    step at which a unit first reaches half its final strength; in brackets, that step over the XOR's")
        for lr, beta2 in ((2e-3, 0.999), (1e-3, 0.999), (5e-4, 0.999), (2.5e-4, 0.999), (1.25e-4, 0.999), (1e-3, 0.99), (1e-3, 0.9)):
            for seed in range(3):
                rows = train('adam', lr, 0.02, seed, int(4000 * 1e-3 / lr), 5, beta2=beta2)
                final = rows[-1][3]
                half = {}
                for j, n in enumerate(NAMES):
                    half[n] = next((r[0] for r in rows if r[3][j] * final[j] > 0 and abs(r[3][j]) >= 0.5 * abs(final[j])), None) if abs(final[j]) >= 0.05 else None
                spikes = sum(1 for a, b in zip(rows, rows[1:]) if b[1] > a[1] + 0.01)
                x = half['xor']
                print(f"  learning rate {lr}, beta2 {beta2}, seed {seed}: " + ', '.join(f"{n} {half[n]} ({half[n] / x:.1f})" if half[n] is not None else f"{n} -" for n in ('xor', 'two back', 'cancel 1', 'cancel 2', 'sure'))
                      + f"; ends at step {rows[-1][0]}, loss {rows[-1][1]:.4f}, {100 * rows[-1][2]:.1f}% explained; {spikes} readouts where the loss rose by over 0.01", end='')
                mid = [r[2] for r in rows if 0.84 < r[1] < 0.90]
                print(f"; explained while the loss is between 0.84 and 0.90: mean {100 * sum(mid) / len(mid):.1f}%, lowest {100 * min(mid):.1f}%", flush=True)
    elif sys.argv[1:] == ['extra']:
        ex = extra_designs()
        print("    share of the logit's variance explained by: the six shapes / plus single violations / plus positions / a lookup table")
        print(f"    the lookup table has {ex[1].max() + 1} entries for {len(rr.P)} prefixes")
        for seed in range(3):
            rows = train('adam', 1e-3, 0.02, seed, 2000, 5, extra=ex)
            print(f"\n  Adam, seed {seed}")
            print(f"  {'step':>6s} {'loss':>7s} {'six':>7s} {'+ violations':>13s} {'+ positions':>12s} {'lookup':>8s}")
            for row in rows[::max(1, len(rows) // 16)] + rows[-1:]:
                print(f"  {row[0]:6d} {row[1]:7.4f} {100 * row[2]:6.1f}% {100 * row[4]:12.1f}% {100 * row[5]:11.1f}% {100 * row[6]:7.1f}%")
            late = [r for r in rows if r[1] < 0.90]
            print("    lowest while the loss is under 0.90: " + ', '.join(f"{name} {100 * min(r[i] for r in late):.1f}%" for name, i in (("six", 2), ("plus violations", 4), ("plus positions", 5), ("lookup", 6))))
            mid = [r for r in rows if 0.84 < r[1] < 0.90]
            print("    mean while the loss is between 0.84 and 0.90: " + ', '.join(f"{name} {100 * sum(r[i] for r in mid) / len(mid):.1f}%" for name, i in (("six", 2), ("plus violations", 4), ("plus positions", 5), ("lookup", 6))), flush=True)
    elif sys.argv[1:] == ['sgd']:
        report("plain SGD, learning rate 0.1, initial std 0.1, seed 0", train('sgd', 0.1, 0.1, 0, 5000, 20))
    else:
        for seed in range(3):
            report(f"Adam, learning rate 0.001, initial std 0.02, seed {seed}", train('adam', 1e-3, 0.02, seed, 2000, 5))
    print(f"    ({time.time() - t0:.0f} s)")
