"""Race between circuits on the self-inoculation toy (https://asving.com/blog/self-inoculation/).

Process: every token is a pair (x, y). The x half comes from HMM1 (Mess3, alpha 0.7, leave 0.03). The
y half comes from one of two Mess3 processes fixed per sequence: situation A (alpha 0.85, leave 0.15)
or situation B (alpha 0.85, leave 0.5). During finetuning the x target has two symbols swapped in
situation A only. Sequences have 24 tokens; 2,500 are sampled per process and binned.

The learner starts from what pretraining would give it, which is NOT trained here:
  tracker            the exact ideal prediction of the next x (as logits)
  situation belief   the exact posterior that the sequence is situation A
  local cue          whether the last two y tokens are equal
  last x token       a weaker vector slot
Candidate units (prices in bits, made up):
  perm(pi, v)        a permutation of the three symbols applied to a vector slot    2 + log2(5) + 3
  gate(s, v)         a scalar slot times a vector slot                              2 + 3 + 3
Readout: logits = fitted linear combination of the tracker and every written slot. Refitting the
strength of a slot that already exists is free.

Flip fraction (the metric of the post): regress the prediction onto the line from the ideal
unswapped prediction (0) to the ideal swapped one (1), on positions 13-24, separately for sequences
from A and from B. A flattened prediction scores 0.5.

The script races single units under the price/benefit rule for two finetuning sets (A and B mixed,
and A only), then turns each resulting chain of circuits into a trajectory over training time by
weighting circuit k by 2^(-price_k - n * loss_k) after n tokens.

Run:  python3 self_inoculation.py     (about ten seconds)
"""
import itertools
import math
import random

random.seed(1)
LN2 = math.log(2)
SLOT = 3.0
L, NS, LATE = 24, 2500, 13
SWAP = (0, 2, 1)  # the finetuning swap: symbols 1 and 2 exchanged


def mess3(alpha, leave):
    beta = (1 - alpha) / 2
    return [[[(1 - 2 * leave if i == j else leave) * (alpha if j == k else beta) for j in range(3)] for i in range(3)] for k in range(3)]


H1, HA, HB = mess3(0.7, 0.03), mess3(0.85, 0.15), mess3(0.85, 0.5)


def sample(TM, n):
    s, out = random.randrange(3), []
    for _ in range(n):
        r, acc, done = random.random(), 0.0, False
        for k in range(3):
            for j in range(3):
                acc += TM[k][s][j]
                if r < acc:
                    done = True
                    break
            if done:
                break
        out.append(k)
        s = j
    return out


def step(f, TM, k):
    return [sum(f[i] * TM[k][i][j] for i in range(3)) for j in range(3)]


# x side: the ideal next-x prediction after each prefix, binned on the simplex (resolution 0.1) with the last token
XH = [dict() for _ in range(L + 1)]
for _ in range(NS):
    seq, f = sample(H1, L), [1 / 3] * 3
    for t, k in enumerate(seq, 1):
        f = step(f, H1, k)
        z = sum(f)
        f = [v / z for v in f]
        p = [sum(f[i] * H1[kk][i][j] for i in range(3) for j in range(3)) for kk in range(3)]
        e = XH[t].setdefault((int(p[0] * 10), int(p[1] * 10), k), [0.0, 0.0, 0.0, 0.0])
        e[0] += 1 / NS
        for i in range(3):
            e[1 + i] += p[i] / NS
# y side: the posterior that the sequence is situation A (binned in steps of 0.05) and the local cue, per situation
YH = {S: [dict() for _ in range(L + 1)] for S in 'AB'}
for S, TMS in (('A', HA), ('B', HB)):
    for _ in range(NS):
        seq, fa, fb, prev = sample(TMS, L), [1 / 6] * 3, [1 / 6] * 3, None
        for t, k in enumerate(seq, 1):
            fa, fb = step(fa, HA, k), step(fb, HB, k)
            z = sum(fa) + sum(fb)
            fa, fb = [v / z for v in fa], [v / z for v in fb]
            key = (min(20, int(sum(fa) * 20 + 0.5)), 1 if prev == k else 0)
            YH[S][t][key] = YH[S][t].get(key, 0.0) + 1 / NS
            prev = k


def table(t_lo, t_hi):
    """Joint cells over the given positions; the x and y halves are independent given the position."""
    cells, span = {}, t_hi - t_lo + 1
    for t in range(t_lo, t_hi + 1):
        for xk, e in XH[t].items():
            for S in 'AB':
                for yk, wy in YH[S][t].items():
                    c = cells.setdefault((xk, yk, S), [0.0, 0.0, 0.0, 0.0])
                    for i in range(4):
                        c[i] += e[i] * wy / span
    out = []
    for (xk, yk, S), c in cells.items():
        p = [c[1] / c[0], c[2] / c[0], c[3] / c[0]]
        z = sum(p)
        out.append({'w': c[0], 'p': [v / z for v in p], 'tok': xk[2], 'a': yk[0] / 20, 'r': float(yk[1]), 'S': S})
    return out


ALL, LAT = table(1, L), table(LATE, L)
PERMS = [p for p in itertools.permutations(range(3)) if p != (0, 1, 2)]


def centred(v):
    m = sum(v) / 3
    return tuple(x - m for x in v)


def feats(c, u):
    if u == 'T':
        return centred([math.log(x) for x in c['p']])
    if u == 'tok':
        return centred([1.0 if c['tok'] == k else 0.0 for k in range(3)])
    v = feats(c, u[2])
    if u[0] == 'perm':
        return tuple(v[u[1][k]] for k in range(3))
    return tuple(c[u[1]] * x for x in v)  # gate


def target(c):
    return [c['p'][SWAP[k]] for k in range(3)] if c['S'] == 'A' else c['p']


def price(u):
    if u in ('T', 'tok'):
        return 0.0
    return 2 + math.log2(5) + SLOT if u[0] == 'perm' else 2 + 2 * SLOT


def solve(A, m):
    for c in range(m):
        piv = max(range(c, m), key=lambda r: abs(A[r][c]))
        A[c], A[piv] = A[piv], A[c]
        for r in range(m):
            if r != c and A[c][c]:
                fac = A[r][c] / A[c][c]
                for cc in range(c, m + 1):
                    A[r][cc] -= fac * A[c][cc]
    return [A[i][m] / A[i][i] if A[i][i] else 0.0 for i in range(m)]


class Task:
    def __init__(self, cells, mix):
        self.cells = [c for c in cells if mix[c['S']] > 0]
        z = sum(c['w'] * mix[c['S']] for c in self.cells)
        self.W = [c['w'] * mix[c['S']] / z for c in self.cells]
        self.Y = [target(c) for c in self.cells]
        self.F = {}

    def f(self, u):
        if u not in self.F:
            self.F[u] = [feats(c, u) for c in self.cells]
        return self.F[u]

    def fit(self, units, iters=14):
        """Softmax regression by Newton's method, starting from the pretrained readout (tracker strength 1).
        With iters=0 it just scores the pretrained readout."""
        Fs = [self.f(u) for u in units]
        m = len(units)
        th = [1.0 if u == 'T' else 0.0 for u in units]
        last = None
        for it in range(iters + 1):
            g = [0.0] * m
            H = [[0.0] * m for _ in range(m)]
            tot = 0.0
            for c in range(len(self.cells)):
                fs = [F[c] for F in Fs]
                z0 = z1 = z2 = 0.0
                for t_, f in zip(th, fs):
                    z0 += t_ * f[0]
                    z1 += t_ * f[1]
                    z2 += t_ * f[2]
                mx = max(z0, z1, z2)
                e0, e1, e2 = math.exp(z0 - mx), math.exp(z1 - mx), math.exp(z2 - mx)
                zs = e0 + e1 + e2
                q0, q1, q2 = e0 / zs, e1 / zs, e2 / zs
                p, w = self.Y[c], self.W[c]
                tot += w * (mx + math.log(zs) - (p[0] * z0 + p[1] * z1 + p[2] * z2))
                av = [q0 * f[0] + q1 * f[1] + q2 * f[2] for f in fs]
                for j in range(m):
                    f = fs[j]
                    g[j] += w * ((q0 - p[0]) * f[0] + (q1 - p[1]) * f[1] + (q2 - p[2]) * f[2])
                    for l in range(m):
                        f2 = fs[l]
                        H[j][l] += w * (q0 * f[0] * f2[0] + q1 * f[1] * f2[1] + q2 * f[2] * f2[2] - av[j] * av[l])
            if it == iters or (last is not None and last - tot < 1e-10):
                last = tot
                break
            last = tot
            d = solve([[H[i][j] + (1e-8 if i == j else 0) for j in range(m)] + [g[i]] for i in range(m)], m)
            th = [t_ - di for t_, di in zip(th, d)]
        return last / LN2, th


def predict(c, units, th):
    z = [0.0, 0.0, 0.0]
    for u, t_ in zip(units, th):
        f = feats(c, u)
        for k in range(3):
            z[k] += t_ * f[k]
    mx = max(z)
    e = [math.exp(v - mx) for v in z]
    zs = sum(e)
    return [v / zs for v in e]


def flip(units, th, cells, S, alo=-1.0, ahi=2.0):
    num = den = 0.0
    for c in cells:
        if c['S'] != S or not (alo <= c['a'] <= ahi):
            continue
        q, p = predict(c, units, th), c['p']
        d = [p[SWAP[k]] - p[k] for k in range(3)]
        num += c['w'] * sum((q[k] - p[k]) * d[k] for k in range(3))
        den += c['w'] * sum(x * x for x in d)
    return num / den if den else float('nan')


def nm(u):
    if u == 'T':
        return 'tracker'
    if u == 'tok':
        return 'last x token'
    if u[0] == 'perm':
        return ('swap' if u[1] == SWAP else f"perm{u[1]}") + f"[{nm(u[2])}]"
    return f"gate[{ {'a': 'situation belief', 'r': 'local cue'}[u[1]] } x {nm(u[2])}]"


def race(name, mix, stages=4):
    task = Task(ALL, mix)
    S = ['T']
    h0, th0 = task.fit(S, iters=0)
    print(f"\n===== finetuning on {name} =====")
    print(f"  pretrained readout:        loss {h0:.4f}   flip in A {flip(S, th0, LAT, 'A'):.2f}, in B {flip(S, th0, LAT, 'B'):.2f}")
    h, th = task.fit(S)
    chain = [(0.0, h, list(S), th)]
    print(f"  tracker strength refit (free): {th[0]:+.2f}   loss {h:.4f}   flip in A {flip(S, th, LAT, 'A'):.2f}, in B {flip(S, th, LAT, 'B'):.2f}")
    for _ in range(stages):
        vecs = ['T', 'tok'] + [u for u in S if u != 'T']
        cand = [('perm', p, v) for v in vecs for p in PERMS] + [('gate', s, v) for v in vecs for s in ('a', 'r')]
        res = []
        for u in cand:
            if u not in S:
                h2, th2 = task.fit(S + [u], iters=10)
                res.append((price(u) / max(h - h2, 1e-9), h - h2, u, h2, th2))
        res.sort(key=lambda r: r[0])
        for r in res[:3]:
            print(f"     candidate: price {price(r[2]):5.2f}  benefit {r[1]:.4f}  price/benefit {r[0]:8.1f}   {nm(r[2])}")
        if res[0][1] < 5e-4:
            print("   nothing substantial left")
            break
        _, b, u, h, th = res[0]
        S = S + [u]
        chain.append((chain[-1][0] + price(u), h, list(S), th))
        print(f"  adopted {nm(u)}:  loss {h:.4f}   flip in A {flip(S, th, LAT, 'A'):.2f}, in B {flip(S, th, LAT, 'B'):.2f}")
        print("     strengths: " + ', '.join(f"{nm(x)} {t:+.2f}" for x, t in zip(S, th)))
    return task, chain


def timeline(chain, grid):
    fl = [(flip(S, th, LAT, 'A'), flip(S, th, LAT, 'B')) for _, _, S, th in chain]
    rows = []
    for n in grid:
        lw = [-k - n * h for k, h, _, _ in chain]
        mx = max(lw)
        ws = [2 ** (x - mx) for x in lw]
        z = sum(ws)
        rows.append((n, sum(w / z * f[0] for w, f in zip(ws, fl)), sum(w / z * f[1] for w, f in zip(ws, fl))))
    return rows


def main():
    print(f"{len(ALL)} cells over all positions, {len(LAT)} over positions {LATE}-{L}")
    task_m, chain_m = race("situations A and B mixed", {'A': 0.5, 'B': 0.5})
    _, chain_a = race("situation A only", {'A': 1.0, 'B': 0.0}, stages=3)
    ideal = sum(w * -sum(y[k] * math.log2(c['a'] * c['p'][SWAP[k]] + (1 - c['a']) * c['p'][k]) for k in range(3))
                for w, y, c in zip(task_m.W, task_m.Y, task_m.cells))
    print(f"\nmixed task: exact ideal loss (belief-weighted mixture of swapped and unswapped) {ideal:.4f}")
    grid = [int(round(10 ** (i / 4))) for i in range(4, 15)]
    print("\ntokens seen     mixed: flip A, flip B     A only: flip A, flip B")
    for (n, a1, b1), (_, a2, b2) in zip(timeline(chain_m, grid), timeline(chain_a, grid)):
        print(f"   {n:6d}            {a1:5.2f}  {b1:5.2f}              {a2:5.2f}  {b2:5.2f}")
    peak = max(timeline(chain_m, [int(1.05 ** i) for i in range(1, 260)]), key=lambda r: r[2])
    print(f"mixed: the flip in B peaks at {peak[2]:.2f} (flip in A {peak[1]:.2f}) after {peak[0]} tokens")
    _, _, Sf, thf = chain_m[-1]
    print("final mixed circuit: flip against the ideal observer's belief that the sequence is A")
    for lo, hi in ((0, 0.1), (0.2, 0.3), (0.45, 0.55), (0.7, 0.8), (0.9, 1.0)):
        fa = [v for v in (flip(Sf, thf, ALL, S_, lo, hi) for S_ in 'AB') if v == v]
        print(f"   belief {lo:.2f}-{hi:.2f}: flip {sum(fa) / len(fa):.2f}")


if __name__ == '__main__':
    main()
