"""The leap staircase of Abbe, Boix-Adsera and Misiakiewicz (2023), Boolean (Fourier-Walsh) case, priced by units.

Setting of the paper: x is uniform on {+1,-1}^d, the target is a sum of monomials in the first few
coordinates, and a two-layer network  f(x) = sum_j a_j s(<w_j, x> + b_j)  with a shifted sigmoid s
is trained on the squared loss. The paper's claim: a monomial that brings in L new coordinates
(a leap of L) takes about d^(L-1) steps of online SGD.

Here a unit is a pair (neuron, monomial). Its dormant factors are the neuron's output weight a_j and
its first-layer weights on the monomial's new coordinates: L + 1 weights at the first stage. Its gain
is the monomial's coefficient times the mean L-th derivative of the activation over the inputs.

  plateau    exact gradients (all 2^d inputs), one monomial of degree L. Plain gradient descent: the end
             of the plateau predicted from the unit's equations, started from the network's own initial
             weights, against the actual run. Then Adam on the same task.
  staircase  exact gradients: z1 + z1 z2 + z1 z2 z3 against z1 z2 z3 alone.
  online     fresh samples every step: steps to learn a monomial against d and against the batch size,
             for plain SGD and for Adam.

Needs numpy.  Run:  python leap.py            (plateau and staircase, about a minute)
                    python leap.py online     (about an hour and a half on one core)
"""
import math
import sys
import time

import numpy as np

SHIFT = 0.5
A0 = 0.01      # initial size of the output weights in the exact-gradient parts: small, so that the network's own
               # output does not disturb the plateau (the paper makes the same choice)


def act(z):
    return 1.0 / (1.0 + np.exp(SHIFT - z))


def act_deriv(z, n):
    s = act(z)
    return (s * (1 - s), s * (1 - s) * (1 - 2 * s), s * (1 - s) * (1 - 6 * s + 6 * s * s),
            s * (1 - s) * (1 - 2 * s) * (1 - 12 * s + 12 * s * s))[n - 1]


def cube(d):
    n = np.arange(2 ** d)[:, None]
    return (1 - 2 * ((n >> np.arange(d)) & 1)).astype(np.float64)


def target(X, monos):
    return sum(c * np.prod(X[:, S], axis=1) for c, S in monos)


class Net:
    def __init__(self, d, M, w0, a0, rng):
        self.W = rng.standard_normal((M, d)) * w0
        self.a = rng.standard_normal(M) * a0
        self.b = np.zeros(M)
        self.params = (self.W, self.a, self.b)

    def out(self, X):
        return act(X @ self.W.T + self.b) @ self.a

    def loss_grad(self, X, y):
        h = act(X @ self.W.T + self.b)
        err = h @ self.a - y
        back = (err[:, None] * self.a) * h * (1 - h)
        n = len(y)
        return 0.5 * float(err @ err) / n, (back.T @ X / n, h.T @ err / n, back.sum(0) / n)


class SGD:
    def __init__(self, params, lr):
        self.p, self.lr = params, lr

    def step(self, grads):
        for p, g in zip(self.p, grads):
            p -= self.lr * g


class Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        c1, c2 = 1 - 0.9 ** self.t, 1 - 0.999 ** self.t
        for p, g, m, v in zip(self.p, grads, self.m, self.v):
            m *= 0.9
            m += 0.1 * g
            v *= 0.999
            v += 0.001 * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + 1e-8)


# ---------------------------------------------------------------- the unit's equations

def predicted(net, X, S, own_output=True, drop=0.95, tmax=1e7):
    """Time at which the unit equations bring the loss 5% under that of predicting zero.

    State: the output weights a (one per neuron) and V, the first-layer weights on the monomial's coordinates.
        da_j/dt  = g_j * prod_i V_ji                      g_j = mean L-th derivative of the activation
        dV_ji/dt = a_j * g_j * prod_{l != i} V_jl
    With own_output, the network's own output is kept to second order in V. It is not part of the target, so
    the loss pulls a and V back. Write s_j for neuron j's activation with its weights on the monomial's coordinates
    set to zero, and K_jl = E[s_j s_l], Q_jl = E[s_j' s_l'], R_jl = E[s_j'' s_l] (primes are derivatives). Then
    (1/2) E f^2 = (1/2) a^T K_V a to second order in V, with K_V = K + Q o (V V^T) + (N R + R^T N)/2 and
    N = diag(|V_j|^2), and the pull-back is gradient descent on that term:
        da -= K_V a,     dV_ji -= a_j (Q diag(a) V)_ji + a_j (R a)_j V_ji."""
    L, n = len(S), len(X)
    W0 = net.W.copy()
    W0[:, S] = 0
    pre = X @ W0.T + net.b
    g = act_deriv(pre, L).mean(0)
    H, H1, H2 = act(pre), act_deriv(pre, 1), act_deriv(pre, 2)
    K, Q, R = H.T @ H / n, H1.T @ H1 / n, H2.T @ H / n
    if not own_output:
        K, Q, R = 0 * K, 0 * Q, 0 * R
    top = max(float(np.linalg.eigvalsh(K).max()), 1e-12)
    a, V, t = net.a.copy(), net.W[:, S].copy(), 0.0
    while t < tmax:
        s = g * V.prod(1)
        sq = (V * V).sum(1)
        Keff = K + Q * (V @ V.T) + 0.5 * R * sq[:, None] + 0.5 * R.T * sq[None, :]
        fhat = float(a @ s)
        if 0.5 * (1 - 2 * fhat + fhat * fhat + float(a @ (Keff @ a))) < drop * 0.5:
            return t
        dV = np.stack([a * g * np.prod(np.delete(V, i, axis=1), axis=1) for i in range(L)], axis=1)
        dV -= a[:, None] * ((Q * a[None, :]) @ V) + (a * (R @ a))[:, None] * V
        dt = 0.02 * float((np.maximum(np.abs(V), 1e-3 * np.abs(V).max()) / (np.abs(dV) + 1e-300)).min())
        dt = min(dt, 0.5 / top, 0.02 * float(np.abs(a).max() / (np.abs(s).max() + 1e-300)) if not own_output else dt)
        a = a + dt * (s - Keff @ a)
        V = V + dt * dV
        t += dt
    return float('inf')


def run_exact(net, X, y, opt, drop=0.95, max_steps=400000):
    """Steps until the exact loss falls below `drop` times the loss of predicting zero."""
    base = 0.5 * float(y @ y) / len(y)
    for s in range(max_steps):
        loss, grads = net.loss_grad(X, y)
        if loss < drop * base:
            return s
        opt.step(grads)
    return None


# ---------------------------------------------------------------- 1. plateau

def part_plateau():
    lr = 0.2
    print("=== plateau: one monomial z1...zL, exact gradients, plain gradient descent ===")
    print("    actual = time (steps x learning rate) at which the loss is 5% under that of predicting zero")
    print("    bare = the (L+1)-factor equations alone; full = the same with the pull-back from the network's own output")
    print(f"    first-layer weights start at the scale shown, output weights at {A0}; nothing is fitted")
    print(f"{'d':>3s} {'neurons':>8s} {'L':>3s} {'scale':>6s} {'seed':>5s} {'actual':>8s} {'bare/actual':>12s} {'full/actual':>12s}")
    summary = {}
    for d, M in ((10, 16), (8, 64)):
        X = cube(d)
        for L, x0 in ((2, 0.1), (3, 0.1), (4, 0.1)):
            S = list(range(L))
            y = target(X, [(1.0, S)])
            for seed in range(3):
                net = Net(d, M, x0, A0, np.random.default_rng(seed))
                bare, full = predicted(net, X, S, own_output=False), predicted(net, X, S)
                steps = run_exact(net, X, y, SGD(net.params, lr), max_steps=int(4 * full / lr) + 2000)
                act_t = steps * lr if steps is not None else float('nan')
                summary.setdefault((M, L), []).append((bare / act_t, full / act_t))
                print(f"{d:3d} {M:8d} {L:3d} {x0:6.2f} {seed:5d} {act_t:8.0f} {bare / act_t:12.2f} {full / act_t:12.2f}", flush=True)
    for (M, L), rows in summary.items():
        print(f"    {M} neurons, L = {L}: bare/actual {min(r[0] for r in rows):.2f} to {max(r[0] for r in rows):.2f}; full/actual {min(r[1] for r in rows):.2f} to {max(r[1] for r in rows):.2f}")

    d, M = 10, 16
    X = cube(d)
    print(f"\n    Adam (learning rate 0.01), d = {d}, {M} neurons: steps until the loss is 5% under that of predicting zero, seeds 0 to 2")
    scales = (0.3, 0.1, 0.03, 0.01)
    print(f"{'L':>3s} " + ' '.join(f"{'scale ' + str(x0):>20s}" for x0 in scales))
    for L in (2, 3, 4):
        S = list(range(L))
        y = target(X, [(1.0, S)])
        row = []
        for x0 in scales:
            res = []
            for seed in range(3):
                net = Net(d, M, x0, A0, np.random.default_rng(seed))
                res.append(run_exact(net, X, y, Adam(net.params, 0.01), max_steps=20000))
            row.append(str(res))
        print(f"{L:3d} " + ' '.join(f"{r:>20s}" for r in row), flush=True)


# ---------------------------------------------------------------- 2. staircase

def part_staircase():
    d, M, x0 = 10, 32, 0.1
    X = cube(d)
    monos = [[0], [0, 1], [0, 1, 2]]
    chi = [np.prod(X[:, S], axis=1) for S in monos]
    print(f"=== staircase: d = {d}, {M} neurons, first-layer weights start at {x0}, output weights at {A0}, exact gradients ===")
    print("    time (steps x learning rate) at which the network's coefficient on each monomial first passes 1/2")
    for name, opt_cls, lr, cap in (("plain gradient descent, learning rate 0.1", SGD, 0.1, 400000), ("Adam, learning rate 0.01", Adam, 0.01, 20000)):
        print(f"  {name}")
        for label, tgt in (("z1 + z1 z2 + z1 z2 z3", [(1.0, S) for S in monos]), ("z1 z2 z3 alone", [(1.0, monos[2])])):
            for seed in range(3):
                net = Net(d, M, x0, A0, np.random.default_rng(seed))
                opt, y = opt_cls(net.params, lr), target(X, tgt)
                want = [i for i, S in enumerate(monos) if any(S == T for _, T in tgt)]
                hit = {}
                for s in range(cap):
                    if s % 5 == 0:
                        f = net.out(X)
                        for i in want:
                            if i not in hit and float(f @ chi[i]) / len(f) > 0.5:
                                hit[i] = s * lr
                        if len(hit) == len(want):
                            break
                    opt.step(net.loss_grad(X, y)[1])
                times = ', '.join(f"{'z1' if i == 0 else 'z1 z2' if i == 1 else 'z1 z2 z3'} at {hit[i]:.1f}" if i in hit else "not reached" for i in want)
                print(f"    {label:24s} seed {seed}: {times}")


# ---------------------------------------------------------------- 3. online

def online(d, L, B, kind, lr, seed, M=64, cap=3000000, drop=0.8):
    """Fresh samples every step. Steps until the loss on a fixed test set is 20% under that of predicting zero."""
    rng = np.random.default_rng(seed)
    net = Net(d, M, 1 / math.sqrt(d), 0.1, rng)
    opt = (SGD if kind == 'sgd' else Adam)(net.params, lr)
    S = list(range(L))
    Xt = rng.integers(0, 2, (4096, d)) * 2.0 - 1
    yt = np.prod(Xt[:, S], axis=1)
    for s in range(cap):
        if s % 100 == 0 and 0.5 * float(np.mean((net.out(Xt) - yt) ** 2)) < drop * 0.5:
            return s
        X = rng.integers(0, 2, (B, d)) * 2.0 - 1
        opt.step(net.loss_grad(X, np.prod(X[:, S], axis=1))[1])
    return None


def slope(xs, ys):
    pts = [(math.log(x), math.log(y)) for x, y in zip(xs, ys) if y]
    if len(pts) < 2:
        return float('nan')
    mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / sum((p[0] - mx) ** 2 for p in pts)


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if len(xs) == 3 else None


def sweep(label, values, run):
    rows = [[run(v, seed) for seed in range(3)] for v in values]
    meds = [med(r) for r in rows]
    print(f"    {label}  " + '   '.join(f"{v}: {r}" for v, r in zip(values, rows)))
    print(f"    {'':{len(label)}s}  medians {meds}; exponent over the last three points {slope(values[-3:], meds[-3:]):.2f}", flush=True)


def part_online():
    print("=== online: fresh samples every step, 64 neurons, first-layer weights of size 1/sqrt(d), output weights 0.1 ===")
    print("    steps until the test loss is 20% under that of predicting zero; three seeds (None = not within 3,000,000 steps)")
    settings = (('sgd', 0.1), ('adam', 0.003))
    for L in (2, 3):
        print(f"  monomial z1...z{L}, batch 32: against the dimension d")
        for kind, lr in settings:
            sweep(f"{kind:4s} lr {lr}", (16, 32, 64, 128, 256), lambda d, seed: online(d, L, 32, kind, lr, seed))
    print("  monomial z1 z2 z3, d = 128: against the batch size")
    for kind, lr in settings:
        sweep(f"{kind:4s} lr {lr}", (8, 32, 128, 512), lambda B, seed: online(128, 3, B, kind, lr, seed))
    print("  monomial z1 z2 z3, Adam, batch 32: larger d")
    sweep("adam lr 0.003", (256, 512, 1024), lambda d, seed: online(d, 3, 32, 'adam', 0.003, seed))


PARTS = {'plateau': part_plateau, 'staircase': part_staircase, 'online': part_online}

if __name__ == '__main__':
    for name in (sys.argv[1:] or ['plateau', 'staircase']):
        t0 = time.time()
        PARTS[name]()
        print(f"    ({time.time() - t0:.0f} s)\n")
