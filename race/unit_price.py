"""What does a composite unit cost? The price is derived from the weights that implement the unit.

A unit is a path of weights that multiply: a head's write, the read of the neuron that takes it,
that neuron's write, and so on. Call the weights still at their small initial size the unit's
dormant factors, k of them, each of size x0. Call the gain g the correlation of the unit's output
(at unit size) with the current prediction error.

  plain gradient descent   each factor grows at g times the product of the others, so the unit
                           reaches full size after  h_k(x0) / g,  where
                               h_1 = 1 - x0,   h_2 = log(1/x0),   h_k = (x0^(2-k) - 1) / (k - 2)
  Adam                     each factor moves about one learning rate per step whatever the size of
                           its gradient, so the wait is a travel distance and does not depend on k or x0

The script checks this four ways.
  chain    a product of k weights fitted to a constant: the two laws in the simplest case
  plateau  the toy transformer of gd_rrxor.py. Its first circuit is a bundle: a head that copies
           earlier tokens (useless alone) and a neuron that reads it. Three dormant factors:
           the head's write omega, the neuron's read v_j, the neuron's write r_j. The predicted
           end of the plateau, with nothing fitted, against the actual run, seed by seed
  adam     the same network trained with Adam: initial scales, seeds, batch sizes
  order    random-start RRXOR (rrxor_random.py): rank every candidate by the signal-to-noise
           ratio of its gradient alone, with no prices

Run:  python3 unit_price.py                (about a minute)
      python3 unit_price.py plateau        (one part: chain, plateau, adam or order)
"""
import math
import random
import sys

import gd_rrxor as gd
from gd_rrxor import Net, L, D, M

FULL = gd.CELLS


def h(k, x0):
    return 1 - x0 if k == 1 else (math.log(1 / x0) if k == 2 else (x0 ** (2 - k) - 1) / (k - 2))


# ---------------------------------------------------------------- 1. a chain of k weights

def chain(k, x0, adam, lr):
    """Fit the product of k equal weights to the constant 1. Time (steps * lr) until the product passes 1/2."""
    x, m, v = x0, 0.0, 0.0
    for s in range(1, 10 ** 7):
        if x ** k >= 0.5:
            return (s - 1) * lr
        g = -(1 - x ** k) * x ** (k - 1)
        if adam:
            m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
            x -= lr * (m / (1 - 0.9 ** s)) / (math.sqrt(v / (1 - 0.999 ** s)) + 1e-8)
        else:
            x -= lr * g


def part_chain():
    x0 = 0.1
    print(f"=== chain: a product of k weights, each starting at {x0}, fitted to the constant 1 ===")
    print("    time = steps x learning rate until the product passes 1/2")
    print(f"{'k':>4s} {'law h_k(x0)':>12s} {'plain gradient descent':>23s} {'Adam':>8s}")
    for k in (1, 2, 3, 4, 6):
        print(f"{k:4d} {h(k, x0):12.1f} {chain(k, x0, False, 0.01):23.1f} {chain(k, x0, True, 0.01):8.2f}")


# ---------------------------------------------------------------- 2. the first unit of the toy transformer

def gains(net):
    """Gain of the soft-XOR bundle through neuron j: E[(y - 1/2) * [neuron j on] * mean of the attended tokens].
    Attention is uniform at the start, and the neuron's on-pattern is the one it has at initialisation."""
    gam = [0.0] * M
    for t, xs, ds, w, ybar in FULL:
        abar = sum(xs[d] for d in ds) / len(ds)
        for j in range(M):
            if net.u[j] * xs[0] + net.gate[j][t] + net.c[j] > 0:
                gam[j] += w * (ybar - 0.5) * abar
    return gam


def blow_up(net, gam, cap=3.0):
    """Time at which  dr_j = gam_j v_j omega,  dv_j = gam_j r_j omega,  domega = sum_j gam_j r_j v_j  leaves the origin."""
    r, v, om, t = list(net.r), list(net.v), net.omega, 0.0

    def f(r, v, om):
        return ([gam[j] * v[j] * om for j in range(M)], [gam[j] * r[j] * om for j in range(M)],
                sum(gam[j] * r[j] * v[j] for j in range(M)))

    gmax = max(abs(g) for g in gam)
    while True:
        amp = max(max(abs(x) for x in r), max(abs(x) for x in v), abs(om))
        if amp > cap or t > 1e8:
            return t
        dt = 0.02 / (gmax * amp)
        dr, dv, dom = f(r, v, om)
        dr, dv, dom = f([r[j] + 0.5 * dt * dr[j] for j in range(M)], [v[j] + 0.5 * dt * dv[j] for j in range(M)], om + 0.5 * dt * dom)
        r = [r[j] + dt * dr[j] for j in range(M)]
        v = [v[j] + dt * dv[j] for j in range(M)]
        om += dt * dom
        t += dt


def part_plateau():
    print("=== plateau: the soft-XOR bundle of the toy transformer (three dormant factors: omega, v_j, r_j) ===")
    print("    predicted = blow-up time of the three-factor equations from the network's own initial weights;")
    print("    actual = time at which gradient descent first brings the loss under 0.95")
    g0 = gains(Net(0.1, 0))
    print("    gains of the six neurons, seed 0: " + ' '.join(f"{g:+.4f}" for g in g0))
    print(f"{'scale':>6s} {'seed':>5s} {'predicted':>10s} {'actual':>8s} {'ratio':>6s}")
    ratios = {}
    for eps, seed in [(0.3, 0), (0.03, 0)] + [(0.1, s) for s in range(7)] + [(0.01, s) for s in range(7)]:
        net = Net(eps, seed)
        pred = blow_up(net, gains(net))
        act = gd.run(eps, seed=seed, stop=0.94)['soft']
        ratios.setdefault(eps, []).append((pred, act))
        print(f"{eps:6.2f} {seed:5d} {pred:10.0f} {act:8.0f} {pred / act:6.2f}")
    for eps in (0.1, 0.01):
        rs = [p / a for p, a in ratios[eps]]
        acts = [a for _, a in ratios[eps]]
        print(f"    scale {eps}: actual times span {min(acts):.0f} to {max(acts):.0f}; predicted/actual between {min(rs):.2f} and {max(rs):.2f}")


# ---------------------------------------------------------------- 3. the same network under Adam

def flat(n):
    return n.b + [n.beta, n.omega] + n.u + n.v + n.c + [x for row in n.gate for x in row] + n.r + [n.rho_x, n.rho_A, n.bias] + n.d


def unflat(n, w):
    it = iter(w)
    for i in range(D):
        n.b[i] = next(it)
    n.beta, n.omega = next(it), next(it)
    for lst in (n.u, n.v, n.c):
        for i in range(M):
            lst[i] = next(it)
    for j in range(M):
        for t in range(L):
            n.gate[j][t] = next(it)
    for i in range(M):
        n.r[i] = next(it)
    n.rho_x, n.rho_A, n.bias = next(it), next(it), next(it)
    for t in range(L):
        n.d[t] = next(it)


def batch_cells(rnd, B):
    cells, w = {}, 1.0 / (B * (L - 1))
    for _ in range(B):
        seq = []
        for p in range(1, L + 1):
            seq.append(seq[-1] ^ seq[-2] if p % 3 == 0 else rnd.getrandbits(1))
        for t in range(1, L):
            key = (t,) + tuple((2 * seq[t - 1 - k] - 1) if t - 1 - k >= 0 else 0 for k in range(D))
            c = cells.setdefault(key, [0.0, 0.0])
            c[0] += w
            c[1] += w * seq[t]
    return [(t, tuple(xs), tuple(d for d in range(D) if t - d >= 1), w, wy / w) for (t, *xs), (w, wy) in cells.items()]


def adam_run(eps, seed, lr=0.01, batch=0, max_steps=4000):
    """Adam on the toy transformer. Returns the steps at which the exact loss first falls under 0.95 and 0.80,
    and the greatest confidence in "next = XOR" reached at non-XOR positions with the confidence at XOR positions then."""
    n, rnd = Net(eps, seed), random.Random(1000 + seed)
    w = flat(n)
    m, v = [0.0] * len(w), [0.0] * len(w)
    soft = gate = None
    peak = (0.0, 0.0)
    try:
        for s in range(1, max_steps + 1):
            gd.CELLS = FULL
            loss, cx, cr, _ = n.step(0.0)
            if soft is None and loss < 0.95:
                soft = s
            if gate is None and loss < 0.80:
                gate = s
                break
            if cr > peak[0]:
                peak = (cr, cx)
            if batch:
                gd.CELLS = batch_cells(rnd, batch)
            n.step(1.0)                                   # a plain step of size 1: the change is minus the gradient
            g = [a - b for a, b in zip(w, flat(n))]
            for i in range(len(w)):
                m[i] = 0.9 * m[i] + 0.1 * g[i]
                v[i] = 0.999 * v[i] + 0.001 * g[i] * g[i]
                w[i] -= lr * (m[i] / (1 - 0.9 ** s)) / (math.sqrt(v[i] / (1 - 0.999 ** s)) + 1e-8)
            unflat(n, w)
    finally:
        gd.CELLS = FULL
    return soft, gate, peak


def part_adam():
    print("=== adam: the same toy transformer trained with Adam (learning rate 0.01, full batch unless stated) ===")
    print(f"{'scale':>7s} {'loss < 0.95 at step':>20s} {'< 0.80 at step':>15s}   P(xor) at XOR positions / elsewhere, at the peak elsewhere")
    for eps in (0.3, 0.1, 0.03, 0.01, 0.001):
        soft, gate, (cr, cx) = adam_run(eps, 0)
        print(f"{eps:7.3f} {soft:20d} {str(gate):>15s}   {cx:.3f} / {cr:.3f}")
    for eps in (0.1, 0.01):
        softs = [adam_run(eps, s)[0] for s in range(7)]
        print(f"    seven seeds at scale {eps}: loss < 0.95 at steps {softs}")
    print("    batch size (sequences per step), scale 0.1, seeds 0 to 2: loss < 0.95 at steps")
    for B in (0, 64, 16, 4, 1):
        print(f"      {('full' if not B else str(B)):>5s}: {[adam_run(0.1, s, batch=B)[0] for s in range(3)]}")


# ---------------------------------------------------------------- 4. random-start RRXOR with no prices

def part_order():
    import rrxor_random as rr
    X, W, Y = rr.X, rr.W, rr.Y

    def factors(us):
        """A rough count of the weights on a path: per unit one write, one read per slot named, one for a position
        pattern other than "all" and one for a sharp pool; plus the final reader's weight."""
        k = 1
        for u in us:
            k += 3 if u[0] == 'gate' else 2 + (1 if u[2] else 0) + (0 if u[1][0] == 'all' else 1) + (0 if (u[4] == 'mean' or u[1][0] == 'off') else 1)
        return k

    def stage(S, extra=()):
        HS, pc = rr.fit(S)
        slots = [X] + list(S)
        R = [w * p - y for w, p, y in zip(W, pc, Y)]
        V = [w * p * (1 - p) for w, p in zip(W, pc)]

        def score(u):
            f = rr.val(u)
            g = sum(r * a for r, a in zip(R, f))
            hh = sum(v * a * a for v, a in zip(V, f))
            return (g * g / hh if hh > 1e-15 else 0.0), abs(g)

        singles = [u for u in rr.gen_units(slots) if u not in S]
        cand = [score(u) + ((u,),) for u in singles]
        for u1 in singles:
            for u2 in rr.gen_units(slots + [u1], must=u1):
                if u2 != u1:
                    cand.append(score(u2) + ((u1, u2),))
        cand.sort(key=lambda c: -c[0])
        print(f"  loss {HS:.4f}; {len(cand)} candidates (single units and linked pairs); best by squared signal-to-noise per token:")
        seen, shown = set(), 0
        for s, _, us in cand:
            col = tuple(rr.val(us[-1]))
            if col in seen:
                continue
            seen.add(col)
            shown += 1
            print(f"    {s:.4f}   " + '  +  '.join(rr.nm(u) for u in us))
            if shown == 3:
                break
        out = []
        for label, us in extra:
            s, g = next((s, g) for s, g, c in cand if c == us)
            above = sum(1 for s2, _, _ in cand if s2 > s + 1e-12)
            print(f"    {s:.4f}   {label}: {above} candidates score higher; gain {g:.4f}, about {factors(us)} factors")
            out.append((g, factors(us)))
        return out

    print("=== order: random-start RRXOR, candidates ranked by the signal-to-noise ratio of their gradient, no prices ===")
    print("    signal = correlation of the candidate's last column with the prediction error; noise = its per-token spread")
    XOR = ('head', ('off', 1), X, X, 'mean')
    VIO = ('head', ('off', 1), X, XOR, 'mean')
    PHASE = ('head', ('per', 3, 2), XOR, VIO, 'max')
    TRICK = ('head', ('off', 2), None, X, 'mean')
    print("STAGE 1")
    stage([], extra=[("the XOR head", (XOR,))])
    rr.NAMES[XOR] = 'XOR'
    print("STAGE 2, holding XOR")
    (gb, kb), (gt, kt) = stage([XOR], extra=[("the reported phase circuit (violation slot + XOR x OR of violations at 2, 5, 8 back)", (VIO, PHASE)),
                                              ("the cheap trick (token two back)", (TRICK,))])
    lo, hi = 0.01, 0.999
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if h(kb, mid) / gb > h(kt, mid) / gt else (lo, mid)
    print(f"  under the multiplying law h_k(x0)/gain, the phase circuit comes before the trick only if x0 > {hi:.2f}")


PARTS = {'chain': part_chain, 'plateau': part_plateau, 'adam': part_adam, 'order': part_order}

if __name__ == '__main__':
    for name in (sys.argv[1:] or list(PARTS)):
        PARTS[name]()
        print()
