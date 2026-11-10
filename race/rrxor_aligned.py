"""Race between circuits on RRXOR with an aligned start.

Process: 12-token sequences  R R X R R X ...  where R is a fair coin and X is the XOR of the two
tokens before it. The learner predicts every next token (log loss, bits per token).
Baseline 1.0; optimum 7/11 = 0.636 (4 of the 11 predictions are XORs).

Stream at position t: the current token x_t = +-1 and a one-hot position.
Units (price in bits, made up):
  copy(d)        attention head writing the token d back                        2 + log2(3) + 3
  match(d)       head whose score is bilinear in query and key: x_t * x_(t-d)   2 + log2(3) + 6
  and(...)       neuron firing on a sign pattern of two slots                   2 + 2*(3+1)
  gate(s, S)     slot s times [position in S]                                   2 + 3 + log2(11) per position
  posb(k)        position bias [t == k]                                         2 + log2(11)
Readout: logit = fitted linear combination of the token and every written slot (strengths are free).

Race: at each stage every single new unit and every pair of new units is scored exactly.
  full attention   a bundle enters after  price / benefit  tokens      (benefit in bits per token)
  search-limited   expected wait  (price + (2^price - 1)/ln 2) / benefit
The bundle with the smallest price/benefit is adopted and the stage repeats.

Run:  python3 rrxor_aligned.py            (about a second)
      python3 rrxor_aligned.py --no-match (attention can only copy; about ten seconds)
"""
import itertools
import math
import sys

LN2 = math.log(2)
L = 12
SLOT, POS, OFF = 3.0, math.log2(L - 1), math.log2(3)


def make_cells():
    """All (position, last four tokens) contexts with their weight and the weight of 'next token is 1'."""
    cells = {}
    free = [p for p in range(1, L + 1) if p % 3]
    for bits in itertools.product((0, 1), repeat=len(free)):
        seq, it = [], iter(bits)
        for p in range(1, L + 1):
            seq.append(seq[-1] ^ seq[-2] if p % 3 == 0 else next(it))
        w = 1.0 / (2 ** len(free) * (L - 1))
        for t in range(1, L):  # tokens 1..t seen, predict token t+1
            key = (t,) + tuple((2 * seq[t - 1 - k] - 1) if t - 1 - k >= 0 else 0 for k in range(4))
            c = cells.setdefault(key, [0.0, 0.0])
            c[0] += w
            c[1] += w * seq[t]
    keys = list(cells)
    return keys, [cells[k][0] for k in keys], [cells[k][1] for k in keys]


KEYS, W, Y = make_cells()
ALLOW_MATCH = '--no-match' not in sys.argv
_val, _loss = {}, {}


def price_u(u):
    kind = u[0]
    if kind == 'gate':
        return 2 + SLOT + POS * len(u[2])
    return {'copy': 2 + OFF + SLOT, 'match': 2 + OFF + 2 * SLOT, 'and': 2 + 2 * (SLOT + 1), 'posb': 2 + POS}[kind]


def price(S):
    return sum(price_u(u) for u in S)


def val(u):
    if u in _val:
        return _val[u]
    if u == 'x0':
        v = [k[1] for k in KEYS]
    elif u[0] == 'copy':
        v = [k[1 + u[1]] for k in KEYS]
    elif u[0] == 'match':
        v = [k[1] * k[1 + u[1]] for k in KEYS]
    elif u[0] == 'and':
        va, vb = val(u[1]), val(u[3])
        v = [1 if (x == u[2] and y == u[4]) else 0 for x, y in zip(va, vb)]
    elif u[0] == 'gate':
        vs = val(u[1])
        v = [x if k[0] in u[2] else 0 for x, k in zip(vs, KEYS)]
    else:  # posb
        v = [1 if k[0] == u[1] else 0 for k in KEYS]
    _val[u] = v
    return v


def solve(A, m):
    """Gauss-Jordan on the augmented m x (m+1) system; returns the solution vector."""
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


def loss(S):
    """Exact expected log loss (bits per token) of the best linear-logit readout over the slots in S."""
    if S in _loss:
        return _loss[S]
    cols = [[1] * len(KEYS), val('x0')] + [val(u) for u in sorted(S, key=repr)]
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
    for _ in range(80):  # damped Newton
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
    _loss[S] = nll(th) / LN2
    return _loss[S]


def moves(S):
    """Every circuit reachable from S by adding one unit (or one position to an existing gate)."""
    out = []
    signed = ['x0'] + sorted([u for u in S if u[0] in ('copy', 'match')], key=repr)
    for d in (1, 2, 3):
        for kind in (('copy', 'match') if ALLOW_MATCH else ('copy',)):
            if (kind, d) not in S:
                out.append(S | {(kind, d)})
    for k in range(1, L):
        if ('posb', k) not in S:
            out.append(S | {('posb', k)})
    for a, b in itertools.combinations(signed, 2):
        for sa in (1, -1):
            for sb in (1, -1):
                u = ('and', a, sa, b, sb)
                if u not in S:
                    out.append(S | {u})
    gates = {u[1]: u for u in S if u[0] == 'gate'}
    for src in ['x0'] + [u for u in S if u[0] in ('copy', 'match', 'and')]:
        if src in gates:
            g = gates[src]
            for k in range(1, L):
                if k not in g[2]:
                    out.append((S - {g}) | {('gate', src, g[2] | {k})})
        else:
            for k in range(1, L):
                out.append(S | {('gate', src, frozenset([k]))})
    return out


def search_time(k, b):
    return (k + (2 ** k - 1) / LN2) / b if b > 1e-9 else float('inf')


def show(u):
    if u == 'x0':
        return 'x0'
    if u[0] == 'gate':
        return f"gate({show(u[1])} @pos {sorted(u[2])})"
    if u[0] == 'and':
        return f"and({show(u[1])}={u[2]:+d}, {show(u[3])}={u[4]:+d})"
    return f"{u[0]}({u[1]})"


def diff(S2, S):
    s = ' + '.join(show(u) for u in sorted(S2 - S, key=repr))
    return s + (f"   [replaces {', '.join(show(u) for u in S - S2)}]" if S - S2 else '')


def stage(S):
    HS = loss(S)
    cand = {}
    singles = moves(S)
    for S1 in singles:
        cand[S1] = (price(S1) - price(S), HS - loss(S1))
    best = min((k / b for k, b in cand.values() if b > 1e-6), default=float('inf'))
    for S1 in singles:
        for S2 in moves(S1):
            if S2 in cand or S2 == S:
                continue
            dk = price(S2) - price(S)
            if dk / max(HS - 0.60, 1e-9) > 3 * best:  # cannot beat the best ratio even with the maximum benefit
                continue
            cand[S2] = (dk, HS - loss(S2))
            if cand[S2][1] > 1e-6:
                best = min(best, dk / cand[S2][1])
    ranked = sorted(((k / b, k, b, S2) for S2, (k, b) in cand.items() if b > 1e-4), key=lambda r: r[0])
    return ranked


def main():
    S = frozenset()
    print(f"baseline loss {loss(S):.4f}, optimum {7 / 11:.4f} bits per token")
    print(f"{'stage':5s} {'adopted':52s} {'price':>6s} {'benefit':>8s} {'tokens (full attn)':>19s} {'tokens (search)':>16s} {'loss':>7s}")
    for st in range(1, 12):
        ranked = stage(S)
        if not ranked or ranked[0][2] < 2e-4:
            break
        ratio, k, b, S2 = ranked[0]
        print(f"{st:5d} {diff(S2, S):52s} {k:6.2f} {b:8.4f} {ratio:19.0f} {search_time(k, b):16.0f} {loss(S2):7.4f}")
        if st <= 2:
            for r2, k2, b2, S3 in ranked[1:4]:
                print(f"      runner-up: {diff(S3, S):41s} {k2:6.2f} {b2:8.4f} {r2:19.0f} {search_time(k2, b2):16.0f}")
        S = S2
        if loss(S) < 7 / 11 + 1e-3:
            break
    print(f"final loss {loss(S):.4f}; circuit: {', '.join(show(u) for u in sorted(S, key=repr))}")
    if ALLOW_MATCH:
        m1 = ('match', 1)
        soft = frozenset([m1])
        full = frozenset([m1, ('gate', m1, frozenset([2, 5, 8, 11]))])
        H0 = loss(frozenset())
        bs, bf = H0 - loss(soft), H0 - loss(full)
        print("\nsoft XOR alone versus the whole gated circuit entered as one bundle (full attention):")
        print(f"  soft XOR: {price(soft):.1f} bits / {bs:.4f} = {price(soft) / bs:.0f} tokens;  gated circuit: {price(full):.1f} / {bf:.4f} = {price(full) / bf:.0f} tokens")
        print(f"  the gated circuit enters first iff the gating costs less than {(bf / bs - 1) * price(soft):.1f} bits (here {price(full) - price(soft):.1f})")


if __name__ == '__main__':
    main()
