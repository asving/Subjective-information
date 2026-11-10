"""Race between circuits on Mess3: does the learner build a belief state?

Process: Mess3, a three-state hidden Markov model over three tokens with x = 0.05 and alpha = 0.85:
the hidden state stays with probability 0.9, and the token names the new state with probability 0.85.
All 3,279 prefixes up to length 7 are enumerated with their exact belief (posterior over the hidden
state) and exact next-token distribution. Uniform guess 1.585 bits per token; optimum 1.163.

Tokens are centred one-hot vectors. One unit type plus an elementwise positive/negative part
(prices in bits, made up):
  head(pattern, q, k, v)   out_t = weighted mean over s in pattern of <q_t, k_s> * v_s   (q, k omitted: plain mean)
        pattern  here, 1/2/3 back, all earlier, all including here, or recency-weighted with decay 0.5/0.7/0.85
        price    2 + pattern (log2 3 for an offset, log2 3 + 1 for a decay, 0 otherwise) + 3 per slot named
  part(slot, sign)         elementwise max(0, +-slot), re-centred                         price 2 + 3 + 1
Readout: logits = fitted linear combination of the written slots (softmax over the three tokens).

Race: single units only, adopting the smallest price/benefit. After each adoption the script reports
the loss and how much of the exact belief's variance a linear read of the slots recovers.

Run:  python3 mess3.py     (about fifteen seconds)
"""
import math
import time

LN2 = math.log(2)
SLOT = 3.0
LMAX = 7
XX, AL = 0.05, 0.85
BE, YY = (1 - AL) / 2, 1 - 2 * XX
TM = [[[(YY if i == j else XX) * (AL if j == k else BE) for j in range(3)] for i in range(3)] for k in range(3)]

CELLS, IDX = [], {}


def enumerate_prefixes(prefix, b, prob):
    if prefix:
        p = [sum(b[i] * TM[k][i][j] for i in range(3) for j in range(3)) for k in range(3)]
        IDX[prefix] = len(CELLS)
        CELLS.append((prefix, b, p, prob))
    if len(prefix) == LMAX:
        return
    for k in range(3):
        nb = [sum(b[i] * TM[k][i][j] for i in range(3)) for j in range(3)]
        s = sum(nb)
        enumerate_prefixes(prefix + (k,), [v / s for v in nb], prob * s)


enumerate_prefixes((), [1 / 3] * 3, 1.0)
C = len(CELLS)
W = [c[3] / LMAX for c in CELLS]
BEL = [c[1] for c in CELLS]
PNEXT = [c[2] for c in CELLS]
ANC = [[IDX[c[0][:s]] for s in range(1, len(c[0]) + 1)] for c in CELLS]
LEN = [len(c[0]) for c in CELLS]
OPT = sum(w * -sum(p * math.log2(p) for p in pn) for w, pn in zip(W, PNEXT))

X = ('x',)
_val = {X: [tuple((1.0 if c[0][-1] == k else 0.0) - 1 / 3 for k in range(3)) for c in CELLS]}
_wts = {}
NAMES = {X: 'token'}
DECAYS = [('dec', 0.5), ('dec', 0.7), ('dec', 0.85)]
PNQ = [('off', 1), ('off', 2), ('off', 3), ('prev',), ('all',)] + DECAYS
PQ = [('off', 0)] + PNQ


def weights(pat, t):
    if pat[0] == 'off':
        return [(t - pat[1], 1.0)] if t - pat[1] >= 1 else []
    if pat[0] == 'prev':
        return [(s, 1.0 / (t - 1)) for s in range(1, t)]
    if pat[0] == 'all':
        return [(s, 1.0 / t) for s in range(1, t + 1)]
    ws = [pat[1] ** (t - s) for s in range(1, t + 1)]
    z = sum(ws)
    return [(s, w / z) for s, w in zip(range(1, t + 1), ws)]


def val(u):
    if u in _val:
        return _val[u]
    out = []
    if u[0] == 'part':
        for f in val(u[1]):
            r = [max(0.0, u[2] * a) for a in f]
            mu = sum(r) / 3
            out.append(tuple(a - mu for a in r))
    else:
        _, pat, q, k, v = u
        V = val(v)
        Q = val(q) if q else None
        K = val(k) if q else None
        for c in range(C):
            t = LEN[c]
            if (pat, t) not in _wts:
                _wts[(pat, t)] = weights(pat, t)
            a0 = a1 = a2 = 0.0
            for s, w in _wts[(pat, t)]:
                j = ANC[c][s - 1]
                vs = V[j]
                if Q:
                    qt, ks = Q[c], K[j]
                    w = w * (qt[0] * ks[0] + qt[1] * ks[1] + qt[2] * ks[2])
                a0 += w * vs[0]
                a1 += w * vs[1]
                a2 += w * vs[2]
            out.append((a0, a1, a2))
    _val[u] = out
    return out


def price_u(u):
    if u[0] == 'part':
        return 2 + SLOT + 1
    pat = u[1]
    pp = 0.0 if pat in (('off', 0), ('prev',), ('all',)) else (math.log2(3) if pat[0] == 'off' else math.log2(3) + 1)
    return 2 + pp + SLOT * (3 if u[2] else 1)


def gen(slots):
    out = [('head', ('off', 0), None, None, X)]
    for pat in PNQ:
        for v in slots:
            out.append(('head', pat, None, None, v))
    for pat in PQ:
        for q in slots:
            for k in slots:
                for v in slots:
                    out.append(('head', pat, q, k, v))
    for s in slots:
        for sg in (1, -1):
            out.append(('part', s, sg))
    return out


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


def fit(units, th0=None, iters=12):
    """Softmax regression by Newton's method: (loss in bits, strengths, per-cell predicted distribution)."""
    Fs = [val(u) for u in units]
    m = len(units)
    th = (list(th0) + [0.0] * m)[:m] if th0 else [0.0] * m
    last, Qs = None, []
    for _ in range(iters):
        g = [0.0] * m
        H = [[0.0] * m for _ in range(m)]
        tot, Qs = 0.0, []
        for c in range(C):
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
            p, w = PNEXT[c], W[c]
            tot += w * (mx + math.log(zs) - (p[0] * z0 + p[1] * z1 + p[2] * z2))
            Qs.append((q0, q1, q2))
            av = [q0 * f[0] + q1 * f[1] + q2 * f[2] for f in fs]
            for j in range(m):
                f = fs[j]
                g[j] += w * ((q0 - p[0]) * f[0] + (q1 - p[1]) * f[1] + (q2 - p[2]) * f[2])
                for l in range(m):
                    f2 = fs[l]
                    H[j][l] += w * (q0 * f[0] * f2[0] + q1 * f[1] * f2[1] + q2 * f[2] * f2[2] - av[j] * av[l])
        if last is not None and last - tot < 1e-9:
            break
        last = tot
        d = solve([[H[i][j] + (1e-9 if i == j else 0) for j in range(m)] + [g[i]] for i in range(m)], m)
        th = [t_ - di for t_, di in zip(th, d)]
    return last / LN2, th, Qs


def belief_r2(units):
    """Share of the belief's variance recovered by a linear read (one coefficient per slot), overall and by position."""
    Fs = [val(u) for u in units]
    m = len(units)
    tgt = [tuple(b - 1 / 3 for b in bel) for bel in BEL]

    def ip(A, B):
        return sum(w * (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) for w, a, b in zip(W, A, B))

    co = solve([[ip(Fs[i], Fs[j]) + (1e-12 if i == j else 0) for j in range(m)] + [ip(Fs[i], tgt)] for i in range(m)], m)
    num = [0.0] * (LMAX + 1)
    den = [0.0] * (LMAX + 1)
    for c in range(C):
        pred = [sum(co[j] * Fs[j][c][k] for j in range(m)) for k in range(3)]
        num[LEN[c]] += W[c] * sum((a - b) ** 2 for a, b in zip(pred, tgt[c]))
        den[LEN[c]] += W[c] * sum(a * a for a in tgt[c])
    return 1 - sum(num) / sum(den), [1 - num[t] / den[t] for t in range(1, LMAX + 1)]


def nm(u):
    if u in NAMES:
        return NAMES[u]
    if u[0] == 'part':
        return f"{'pos' if u[2] == 1 else 'neg'}-part[{nm(u[1])}]"
    pat = u[1]
    ps = {('off', 0): 'here', ('off', 1): '1 back', ('off', 2): '2 back', ('off', 3): '3 back', ('prev',): 'all earlier',
          ('all',): 'all incl. here'}.get(pat) or f"recency-weighted, decay {pat[1]}"
    if not u[2]:
        return f"copy[{nm(u[4])}; {ps}]"
    return f"match[{ps}: <{nm(u[2])} here, {nm(u[3])} there> x {nm(u[4])} there]"


def main():
    print(f"{C} prefixes up to length {LMAX}; uniform guess {math.log2(3):.4f}, exact optimum {OPT:.4f} bits per token")
    S, th, HS = [], [], math.log2(3)
    Qs = [(1 / 3, 1 / 3, 1 / 3)] * C
    for st in range(1, 7):
        t0 = time.time()
        cand = []
        for u in gen([X] + S):
            if u in S:
                continue
            f = val(u)
            g = h = 0.0
            for c in range(C):  # one-step estimate of the gain from adding this slot
                q, p, fc = Qs[c], PNEXT[c], f[c]
                a = q[0] * fc[0] + q[1] * fc[1] + q[2] * fc[2]
                g += W[c] * ((q[0] - p[0]) * fc[0] + (q[1] - p[1]) * fc[1] + (q[2] - p[2]) * fc[2])
                h += W[c] * (q[0] * fc[0] ** 2 + q[1] * fc[1] ** 2 + q[2] * fc[2] ** 2 - a * a)
            gain = (g * g / (2 * h) / LN2) if h > 1e-14 else 0.0
            if gain > 1e-5:
                cand.append((price_u(u) / gain, u))
        cand.sort(key=lambda c: c[0])
        exact = []
        for _, u in cand[:6]:
            h2, th2, q2 = fit(S + [u], th, iters=8)
            exact.append((price_u(u) / max(HS - h2, 1e-9), price_u(u), HS - h2, u, th2, q2, h2))
        exact.sort(key=lambda e: e[0])
        if not exact or exact[0][2] < 5e-4:
            print("nothing substantial left")
            break
        _, pr, b, u, th, Qs, HS = exact[0]
        desc = nm(u)
        S.append(u)
        NAMES[u] = f"U{len(S)}"
        r2, r2pos = belief_r2(S)
        print(f"stage {st}: U{len(S)} = {desc}")
        print(f"         price {pr:.2f}  benefit {b:.4f}  loss {HS:.4f} (gap {HS - OPT:.4f})  belief R2 {r2:.4f}  by position " + ' '.join(f"{v:.2f}" for v in r2pos)
              + f"   ({len(cand)} candidates, {time.time() - t0:.0f}s)")
    print("readout strengths: " + ', '.join(f"U{i + 1} {t:+.2f}" for i, t in enumerate(th)))
    print("a single recency-weighted token tally on its own:")
    for lam in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        u = ('head', ('dec', lam), None, None, X)
        h, _, _ = fit([u], iters=8)
        print(f"   decay {lam}: loss {h:.4f}, belief R2 {belief_r2([u])[0]:.4f}")


if __name__ == '__main__':
    main()
