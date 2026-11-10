"""Race between circuits on RRXOR with a random start (the phase is unknown).

Process: R R X R R X ... as in rrxor_aligned.py, but each 12-token sequence starts at the first R,
the second R or the X with probability 1/3 each. The optimal predictor has to infer the phase from
the history. Baseline 1.0 bits per token; exact optimum 0.8234.

Stream at position t: the current token x_t = +-1. One unit type plus a gate (prices in bits, made up):
  head(pattern, q, k, pool)   out_t = q_t * pool over s in pattern of k_s
        pattern  a fixed offset (1, 2 or 3 back), every p-th position back from r (p = 2 or 3), or all
        q        a slot here, or none;  k  a slot there (raw token, or a slot another unit wrote)
        pool     mean, max or min (max over a +-1 slot is an OR)
        price    2 + pattern (log2 3 for an offset, 1 + log2 p for a periodic one, 0 for all)
                   + 3 per slot named + 1 for max/min
  gate(src, cond, v)          src * [cond == v]                         price 2 + 2*3 + 1
Readout: logit = fitted linear combination of the token and every written slot.

The script
  1. runs a blind race: every single unit and every linked pair (the second unit reads the first)
     at stages 1 and 2, single units afterwards, adopting the smallest price/benefit;
  2. builds the circuit that trained transformers are reported to find (XOR, then an OR over the
     history of XOR violations, shifted by multiples of 3) and prices each step under both rules;
  3. checks that this circuit carries the exact phase belief.

Run:  python3 rrxor_random.py     (about a minute)
"""
import itertools
import math
import time

LN2 = math.log(2)
L = 12
SLOT = 3.0

# ---- cells: every prefix of a random-phase sequence, with its weight, the weight of "next token is +1",
#      and the weight of each phase relative to the current position ----
pref = {}
for o in (0, 1, 2):
    n = L + o
    free = [p for p in range(1, n + 1) if p % 3]
    for bits in itertools.product((0, 1), repeat=len(free)):
        seq, it = [], iter(bits)
        for p in range(1, n + 1):
            seq.append(seq[-1] ^ seq[-2] if p % 3 == 0 else next(it))
        seq = tuple(2 * b - 1 for b in seq[o:])
        w = 1.0 / (3 * 2 ** len(free) * (L - 1))
        for t in range(1, L):
            c = pref.setdefault(seq[:t], [0.0, 0.0, 0.0, 0.0, 0.0])
            c[0] += w
            c[1] += w * (seq[t] == 1)
            c[2 + (t + 1 + o) % 3] += w  # 0: next token is the XOR, 1: current token is, 2: previous token is
P = list(pref)
IDX = {p: i for i, p in enumerate(P)}
W = [pref[p][0] for p in P]
Y = [pref[p][1] for p in P]
BELIEF = [tuple(x / pref[p][0] for x in pref[p][2:]) for p in P]
ANC = [[IDX[p[:s]] for s in range(1, len(p) + 1)] for p in P]


def ent(q):
    return 0.0 if q <= 0 or q >= 1 else -(q * math.log2(q) + (1 - q) * math.log2(1 - q))


def table_loss(cols):
    """Loss of the best predictor that is an arbitrary function of the given columns."""
    g = {}
    for key, w, y in zip(zip(*cols), W, Y):
        a = g.setdefault(key, [0.0, 0.0])
        a[0] += w
        a[1] += y
    return sum(a[0] * ent(a[1] / a[0]) for a in g.values())


X = ('x',)
PATTERNS = [('off', 1), ('off', 2), ('off', 3), ('per', 2, 0), ('per', 2, 1), ('per', 3, 0), ('per', 3, 1), ('per', 3, 2), ('all',)]
_val = {}
NAMES = {}


def val(u):
    if u in _val:
        return _val[u]
    if u == X:
        v = [p[-1] for p in P]
    elif u[0] == 'head':
        _, pat, q, k, pool = u
        vk = val(k)
        vq = val(q) if q else None
        v = []
        for i, p in enumerate(P):
            t, a = len(p), ANC[i]
            ds = (pat[1],) if pat[0] == 'off' else (range(pat[2], t, pat[1]) if pat[0] == 'per' else range(0, t))
            xs = [z for z in (vk[a[t - d - 1]] for d in ds if t - d >= 1) if z != 0]
            r = 0 if not xs else (max(xs) if pool == 'max' else min(xs) if pool == 'min' else round(sum(xs) / len(xs), 6))
            v.append(r * vq[i] if vq is not None else r)
    else:  # gate
        vs, vc = val(u[1]), val(u[2])
        v = [a if b == u[3] else 0 for a, b in zip(vs, vc)]
    _val[u] = v
    return v


def price_u(u):
    if u[0] == 'head':
        pat = u[1]
        pp = math.log2(3) if pat[0] == 'off' else (1 + math.log2(pat[1]) if pat[0] == 'per' else 0.0)
        return 2 + pp + SLOT + (SLOT if u[2] else 0) + (0 if (u[4] == 'mean' or pat[0] == 'off') else 1)
    return 2 + 2 * SLOT + 1


def gen_units(slots, must=None):
    out = []
    for pat in PATTERNS:
        for q in [None] + slots:
            for k in slots:
                if must and must not in (q, k):
                    continue
                for pool in (('mean',) if pat[0] == 'off' else ('mean', 'max', 'min')):
                    out.append(('head', pat, q, k, pool))
    for src in slots:
        for cond in slots:
            if src == cond or (must and must not in (src, cond)):
                continue
            for cv in (1, -1):
                out.append(('gate', src, cond, cv))
    return out


def solve(A, m):
    for c in range(m):
        piv = max(range(c, m), key=lambda r: abs(A[r][c]))
        A[c], A[piv] = A[piv], A[c]
        if abs(A[c][c]) < 1e-300:
            continue
        for r in range(m):
            if r != c:
                fac = A[r][c] / A[c][c]
                if fac:
                    for cc in range(c, m + 1):
                        A[r][cc] -= fac * A[c][cc]
    return [A[i][m] / A[i][i] if abs(A[i][i]) > 1e-300 else 0.0 for i in range(m)]


def fit(units):
    """Best linear-logit readout over the token and the given units: (loss in bits, per-cell probability)."""
    cols = [[1] * len(P), val(X)] + [val(u) for u in units]
    agg = {}
    for f, w, y in zip(zip(*cols), W, Y):
        a = agg.setdefault(f, [0.0, 0.0])
        a[0] += w
        a[1] += y
    F = list(agg)
    Wa = [agg[f][0] for f in F]
    Ya = [agg[f][1] for f in F]
    m, lam = len(F[0]), 1e-7

    def nll(th):
        s = 0.0
        for f, w, y in zip(F, Wa, Ya):
            z = sum(a * b for a, b in zip(f, th) if a)
            s += w * (max(z, 0) + math.log1p(math.exp(-abs(z)))) - y * z
        return s

    th = [0.0] * m
    cur = nll(th)
    for _ in range(80):
        g = [0.0] * m
        H = [[0.0] * m for _ in range(m)]
        for f, w, y in zip(F, Wa, Ya):
            z = max(-35, min(35, sum(a * b for a, b in zip(f, th) if a)))
            p = 1 / (1 + math.exp(-z))
            r, v = w * p - y, w * p * (1 - p)
            for i in range(m):
                if f[i]:
                    g[i] += r * f[i]
                    for j in range(m):
                        if f[j]:
                            H[i][j] += v * f[i] * f[j]
        for i in range(m):
            g[i] += 2 * lam * th[i]
            H[i][i] += 2 * lam
        d = solve([row + [gi] for row, gi in zip(H, g)], m)
        step = 1.0
        while True:
            new = [t - step * di for t, di in zip(th, d)]
            nl = nll(new) + lam * sum(t * t for t in new)
            if nl <= cur + 1e-15 or step < 1e-5:
                break
            step /= 2
        done = cur - nl < 1e-11
        th, cur = new, nl
        if done:
            break
    pc = [1 / (1 + math.exp(-max(-35, min(35, sum(a * b for a, b in zip(f, th)))))) for f in zip(*cols)]
    return nll(th) / LN2, pc


def search_time(k, b):
    return (k + (2 ** k - 1) / LN2) / b if b > 1e-9 else float('inf')


def nm(u):
    if u is None:
        return '-'
    if u == X:
        return 'x'
    if u in NAMES:
        return NAMES[u]
    if u[0] == 'head':
        pat = u[1]
        ps = f"{pat[1]} back" if pat[0] == 'off' else (f"every {pat[1]} back from {pat[2]}" if pat[0] == 'per' else "all")
        return f"head[{ps}; q={nm(u[2])}; k={nm(u[3])}; {u[4]}]"
    return f"gate[{nm(u[1])} when {nm(u[2])}={u[3]:+d}]"


def stage(S, pairs, top=4, keep=40):
    """Screen every candidate with a one-step estimate, then score the best exactly under both rules."""
    t0 = time.time()
    HS, pc = fit(S)
    slots = [X] + list(S)
    R = [w * p - y for w, p, y in zip(W, pc, Y)]
    V = [w * p * (1 - p) for w, p in zip(W, pc)]

    def approx(us):
        fs = [val(u) for u in us]
        if len(us) == 1:
            f = fs[0]
            g = sum(r * a for r, a in zip(R, f))
            h = sum(v * a * a for v, a in zip(V, f))
            return (g * g / (2 * h) / LN2) if h > 1e-15 else 0.0
        f1, f2 = fs
        g1 = sum(r * a for r, a in zip(R, f1))
        g2 = sum(r * a for r, a in zip(R, f2))
        h11 = sum(v * a * a for v, a in zip(V, f1))
        h22 = sum(v * a * a for v, a in zip(V, f2))
        h12 = sum(v * a * b for v, a, b in zip(V, f1, f2))
        det = h11 * h22 - h12 * h12
        if det < 1e-18:
            return max((g1 * g1 / (2 * h11) if h11 > 1e-15 else 0), (g2 * g2 / (2 * h22) if h22 > 1e-15 else 0)) / LN2
        return (g1 * (h22 * g1 - h12 * g2) + g2 * (h11 * g2 - h12 * g1)) / det / 2 / LN2

    cand = []
    singles = [u for u in gen_units(slots) if u not in S]
    for u in singles:
        cand.append((approx([u]), (u,)))
    npairs = 0
    if pairs:
        for u1 in singles:
            for u2 in gen_units(slots + [u1], must=u1):
                if u2 != u1:
                    npairs += 1
                    cand.append((approx([u1, u2]), (u1, u2)))
    cand = [(g, us, sum(price_u(u) for u in us)) for g, us in cand if g > 1e-5]
    short = {us for _, us, _ in sorted(cand, key=lambda c: c[2] / c[0])[:keep]}
    short |= {us for _, us, _ in sorted(cand, key=lambda c: search_time(c[2], c[0]))[:keep]}
    exact = []
    for us in short:
        h, _ = fit(list(S) + list(us))
        exact.append((sum(price_u(u) for u in us), HS - h, us))
    exact = [e for e in exact if e[1] > 1e-5]
    by_ratio = sorted(exact, key=lambda e: e[0] / e[1])
    by_search = sorted(exact, key=lambda e: search_time(e[0], e[1]))
    print(f"  loss {HS:.4f}; {len(singles)} single and {npairs} paired candidates ({time.time() - t0:.0f}s)")
    for title, lst in (("full attention (price/benefit)", by_ratio), ("search-limited", by_search)):
        print(f"   best under {title}:")
        for pr, b, us in lst[:top]:
            print(f"     price {pr:5.2f}  benefit {b:.4f}  tokens: full {pr / b:8.0f}  search {search_time(pr, b):12.0f}   " + '  +  '.join(nm(u) for u in us))
    return by_ratio


def main():
    print(f"{len(P)} distinct prefixes; exact optimum {table_loss([P]):.4f} bits per token")
    print("\n== blind race ==")
    print("STAGE 1 (single units and linked pairs)")
    ranked = stage([], True)
    S = list(ranked[0][2])
    NAMES[S[0]] = 'XOR'
    print("STAGE 2, holding XOR (single units and linked pairs)")
    ranked = stage(S, True, top=6)
    for st in range(3, 8):
        if not ranked or ranked[0][1] < 3e-4:
            print("   nothing substantial left")
            break
        for u in ranked[0][2]:
            S.append(u)
            NAMES[u] = f"U{len(S)}"
            print(f"   adopted U{len(S)} = {u}")
        print(f"STAGE {st} (single units only)")
        ranked = stage(S, False, top=2)
    print(f"blind race ends at loss {fit(S)[0]:.4f} with {len(S)} units, {sum(price_u(u) for u in S):.1f} bits")

    print("\n== the reported circuit, built by hand ==")
    XOR = ('head', ('off', 1), X, X, 'mean')
    VIO = ('head', ('off', 1), X, XOR, 'mean')  # violation at s: token s times the XOR written at s-1
    DEAD = {r: ('head', ('per', 3, r), None, VIO, 'max') for r in (0, 1, 2)}  # any violation at offsets r, r+3, ...
    F = {r: ('head', ('per', 3, r), XOR, VIO, 'max') for r in (0, 1, 2)}  # the same, multiplied by the XOR here
    h_xor = fit([XOR])[0]
    print(f"   violation slot alone: benefit {h_xor - fit([XOR, VIO])[0]:.5f}")
    for r, lab in ((2, 'offsets 2,5,8 (the next token)'), (0, 'offsets 0,3,6,9 (the current token)'), (1, 'offsets 1,4,7,10 (the previous token)')):
        b = h_xor - fit([XOR, VIO, F[r]])[0]
        pr = price_u(VIO) + price_u(F[r])
        print(f"   violation slot + XOR x OR of violations at {lab:38s} price {pr:.2f}  benefit {b:.4f}  tokens: full {pr / b:6.0f}  search {search_time(pr, b):12.0f}")
    steps = [("XOR", [XOR]), ("+ violation, XOR x OR at 2,5,8", [XOR, VIO, F[2]]), ("+ XOR x OR at 0,3,6,9", [XOR, VIO, F[2], F[0]]),
             ("+ XOR x OR at 1,4,7,10", [XOR, VIO, F[2], F[0], F[1]]),
             ("+ conjunction (both other phases dead)", [XOR, VIO, F[2], F[0], F[1], DEAD[1], ('gate', F[0], DEAD[1], 1)])]
    for name, us in steps:
        print(f"   {name:42s} loss {fit(us)[0]:.4f}   circuit price {sum(price_u(u) for u in us):5.1f} bits")
    xm, dA, dB, dC = val(XOR), val(DEAD[2]), val(DEAD[0]), val(DEAD[1])
    tmod = [len(p) % 3 for p in P]
    print("   as a lookup table:  XOR only %.4f;  XOR + three OR flags %.4f;  + position mod 3 %.4f" % (
        table_loss([xm]), table_loss([xm, dA, dB, dC]), table_loss([xm, dA, dB, dC, tmod])))

    print("\n== the phase belief ==")
    for label, key in (("three OR flags", lambda i: (dA[i], dB[i], dC[i])), ("three OR flags + position mod 3", lambda i: (dA[i], dB[i], dC[i], tmod[i]))):
        g = {}
        for i in range(len(P)):
            g.setdefault(key(i), []).append(BELIEF[i])
        worst = max(max(abs(a - b) for v in vs for a, b in zip(v, vs[0])) for vs in g.values())
        print(f"   belief as a function of the {label}: {len(g)} states, largest disagreement inside a state {worst:.2e}")
    print(f"   distinct beliefs over (next is XOR, current is XOR, previous is XOR): {len(set(tuple(round(x, 6) for x in b) for b in BELIEF))}")


if __name__ == '__main__':
    main()
