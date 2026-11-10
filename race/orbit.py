"""Race between circuits on orbits of a random orthogonal matrix.

Process: x_0 = v, x_1 = Uv, x_2 = U^2 v, ...  with v a random unit vector and U a random orthogonal
matrix (either determinant), both redrawn for every sequence. Dimension 4, ten tokens, 300 sampled
sequences. The learner predicts each next token; loss is squared error, so predicting zero scores 1.
The exact optimum fits x_t as a linear combination of the earlier tokens and applies the same
coefficients to the shifted tokens; it is exact once the context spans the space.

One unit type (price in bits, made up):
  head(pattern, q, k, v)   out_t = mean over s in pattern of <q_t, k_s> * v_s     (q, k omitted: plain mean of v_s)
        pattern  here, 1/2/3 back, all earlier, all including here
        q, k, v  vector slots: the raw token or a slot another unit wrote
        price    2 + log2(3) for a nonzero offset + 3 per slot named
Readout: prediction = fitted linear combination of the written slots.

Race: single units and all pairs at stage 1, single units afterwards; the smallest price/benefit is
adopted (benefit = drop in mean squared error).

Run:  python3 orbit.py     (about ten seconds; numbers carry sampling noise of 0.01-0.02)
"""
import math
import random
import time
from operator import mul

random.seed(0)
LN2 = math.log(2)
SLOT = 3.0
D, T, N = 4, 10, 300


def dot(a, b):
    return sum(map(mul, a, b))


def haar(d):
    """Gram-Schmidt on Gaussian vectors: a Haar-distributed orthogonal matrix."""
    cols = []
    while len(cols) < d:
        v = [random.gauss(0, 1) for _ in range(d)]
        for c in cols:
            p = dot(v, c)
            v = [a - p * b for a, b in zip(v, c)]
        nrm = math.sqrt(dot(v, v))
        cols.append([a / nrm for a in v])
    return cols


DATA = []
for _ in range(N):
    U = haar(D)
    v = [random.gauss(0, 1) for _ in range(D)]
    nrm = math.sqrt(dot(v, v))
    x = tuple(a / nrm for a in v)
    seq = [x]
    for _ in range(T - 1):
        x = tuple(sum(U[j][i] * x[j] for j in range(D)) for i in range(D))
        seq.append(x)
    DATA.append(seq)
ZERO = tuple([0.0] * D)
TP = T - 1  # prediction positions t = 0..T-2, target x_(t+1)
YT = [[DATA[n][t + 1] for t in range(TP)] for n in range(N)]
SYY = sum(dot(y, y) for row in YT for y in row)

X = ('x',)
_val = {X: [[DATA[n][t] for t in range(TP)] for n in range(N)]}
PQ = [('off', 0), ('off', 1), ('off', 2), ('off', 3), ('prev',), ('all',)]
PN = PQ[1:]
NAMES = {X: 'x'}
_gram, _b = {}, {}


def optimum():
    """Exact optimum by position: the part of x_t outside span(x_0..x_(t-1)) goes to an unknown direction."""
    opt = [0.0] * TP
    for n in range(N):
        basis = []
        for t in range(TP):
            r = list(DATA[n][t])
            for b in basis:
                p = dot(r, b)
                r = [a - p * c for a, c in zip(r, b)]
            nr2 = dot(r, r)
            opt[t] += nr2 / N
            if nr2 > 1e-18:
                basis.append([a / math.sqrt(nr2) for a in r])
    return opt


def val(u):
    if u in _val:
        return _val[u]
    _, pat, q, k, vv = u
    V = val(vv)
    Q = val(q) if q else None
    K = val(k) if q else None
    out = []
    for n in range(N):
        Vn, row = V[n], []
        for t in range(TP):
            if pat[0] == 'off':
                ss = (t - pat[1],) if t - pat[1] >= 0 else ()
            else:
                ss = range(0, t) if pat[0] == 'prev' else range(0, t + 1)
            acc, cnt = [0.0] * D, 0
            for s in ss:
                w = dot(Q[n][t], K[n][s]) if Q else 1.0
                vs = Vn[s]
                cnt += 1
                for i in range(D):
                    acc[i] += w * vs[i]
            row.append(tuple(a / cnt for a in acc) if cnt else ZERO)
        out.append(row)
    _val[u] = out
    return out


def price_u(u):
    pp = math.log2(3) if u[1][0] == 'off' and u[1][1] > 0 else 0.0
    return 2 + pp + SLOT * (3 if u[2] else 1)


def gen(slots, must=None):
    out = []
    for pat in PN:
        for v in slots:
            if not must or v == must:
                out.append(('head', pat, None, None, v))
    for pat in PQ:
        for q in slots:
            for k in slots:
                for v in slots:
                    if not must or must in (q, k, v):
                        out.append(('head', pat, q, k, v))
    return out


def gram(u, v):
    key = (u, v) if repr(u) <= repr(v) else (v, u)
    if key not in _gram:
        A, C = val(u), val(v)
        _gram[key] = sum(dot(a, c) for ra, rc in zip(A, C) for a, c in zip(ra, rc))
    return _gram[key]


def bvec(u):
    if u not in _b:
        A = val(u)
        _b[u] = sum(dot(a, y) for ra, ry in zip(A, YT) for a, y in zip(ra, ry))
    return _b[u]


def fit(units):
    """Least squares over the written slots: (mean squared error as a fraction of baseline, coefficients)."""
    m = len(units)
    if m == 0:
        return 1.0, []
    A = [[gram(units[i], units[j]) + (1e-9 * SYY if i == j else 0) for j in range(m)] + [bvec(units[i])] for i in range(m)]
    for c in range(m):
        piv = max(range(c, m), key=lambda r: abs(A[r][c]))
        A[c], A[piv] = A[piv], A[c]
        for r in range(m):
            if r != c and A[c][c]:
                fac = A[r][c] / A[c][c]
                for cc in range(c, m + 1):
                    A[r][cc] -= fac * A[c][cc]
    th = [A[i][m] / A[i][i] if A[i][i] else 0.0 for i in range(m)]
    return (SYY - sum(t * bvec(u) for t, u in zip(th, units))) / SYY, th


def by_position(units):
    _, th = fit(units)
    Fs = [val(u) for u in units]
    out = [0.0] * TP
    for n in range(N):
        for t in range(TP):
            y = YT[n][t]
            pred = [sum(th[j] * Fs[j][n][t][i] for j in range(len(units))) for i in range(D)]
            out[t] += sum((a - b) ** 2 for a, b in zip(y, pred)) / N
    return out


def nm(u):
    if u in NAMES:
        return NAMES[u]
    ps = {('off', 0): 'here', ('off', 1): '1 back', ('off', 2): '2 back', ('off', 3): '3 back', ('prev',): 'all earlier', ('all',): 'all incl. here'}[u[1]]
    if not u[2]:
        return f"copy[{nm(u[4])} from {ps}]"
    return f"match[over {ps}: <{nm(u[2])} here, {nm(u[3])} there> x {nm(u[4])} there]"


def search_time(k, b):
    return (k + (2 ** k - 1) / LN2) / b if b > 1e-9 else float('inf')


def stage(S, pairs, top=3):
    t0 = time.time()
    HS, _ = fit(S)
    slots = [X] + S
    cand = []
    singles = [u for u in gen(slots) if u not in S]
    for u in singles:
        cand.append((price_u(u), HS - fit(S + [u])[0], (u,)))
    npair = 0
    if pairs:
        for u1 in singles:
            for u2 in gen(slots + [u1], must=u1):
                if u2 != u1:
                    npair += 1
                    cand.append((price_u(u1) + price_u(u2), HS - fit(S + [u1, u2])[0], (u1, u2)))
    cand = [c for c in cand if c[1] > 1e-4]
    print(f"  MSE {HS:.4f}; {len(singles)} single and {npair} paired candidates ({time.time() - t0:.0f}s)")
    for title, lst in (("full attention (price/benefit)", sorted(cand, key=lambda c: c[0] / c[1])),
                       ("search-limited", sorted(cand, key=lambda c: search_time(c[0], c[1])))):
        print(f"   best under {title}:")
        for pr, b, us in lst[:top]:
            print(f"     price {pr:5.2f}  benefit {b:.4f}  ratio {pr / b:7.1f}  search {search_time(pr, b):9.0f}   " + '  +  '.join(nm(u) for u in us))
    return sorted(cand, key=lambda c: c[0] / c[1])


def main():
    opt = optimum()
    print(f"dimension {D}, {T} tokens, {N} sequences")
    print("optimum MSE by position: " + ' '.join(f"{o:.2f}" for o in opt) + f"   mean {sum(opt) / TP:.4f}")
    print("STAGE 1 (single units and pairs)")
    ranked = stage([], True, top=4)
    S = list(ranked[0][2])
    for i, u in enumerate(S):
        NAMES[u] = f"U{i + 1}"
        print(f"   adopted U{i + 1} = {u}")
    print("   MSE by position: " + ' '.join(f"{v:.2f}" for v in by_position(S)))
    for st in range(2, 5):
        print(f"STAGE {st} (single units only)")
        ranked = stage(S, False, top=2)
        if not ranked or ranked[0][1] < 2e-3:
            print("   nothing substantial left among single units")
            break
        u = ranked[0][2][0]
        S.append(u)
        NAMES[u] = f"U{len(S)}"
        print(f"   adopted U{len(S)}: reads {', '.join(NAMES[s] for s in u[2:] if s in NAMES)} over {u[1]}")
    print(f"final MSE {fit(S)[0]:.4f} (optimum {sum(opt) / TP:.4f}); by position: " + ' '.join(f"{v:.2f}" for v in by_position(S)))


if __name__ == '__main__':
    main()
