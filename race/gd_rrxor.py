"""Gradient descent on a one-layer transformer whose units are attention components and MLP neurons.

Process: RRXOR with an aligned start, as in rrxor_aligned.py (12 tokens, exact expectation over all
sequences). Baseline 1.0 bits per token, soft XOR 0.892, optimum 0.636.

The stream at position t has columns: the token x_t = +-1, a one-hot position, the attention output A,
and one column per neuron. Every weight below is one scalar.

  attention (one head)
      score      s(t, t') = b[t - t'] + beta * x_t * x_t'        for t' = t, t-1, t-2, t-3
                 (b: relative position bias, one unit per offset; beta: one content component)
      weights    alpha(t, .) = softmax of the scores               the only pool is this weighted mean
      output     A_t = omega * sum over t' of alpha(t, t') * x_t'  (omega: one value/output component)
  MLP neurons    n_j = ReLU(u_j * x_t + v_j * A_t + g_j[t] + c_j)   (g_j[t]: the neuron's read of position t)
  unembedding    logit_t = sum_j r_j * n_j + rho_x * x_t + rho_A * A_t + d[t] + bias

All weights start as independent Gaussians of standard deviation eps and follow plain full-batch
gradient descent on the exact expected log loss. The script prints, over training time (steps times
learning rate), the loss, how confidently the network predicts "next = XOR of the last two tokens"
at XOR positions and at the other positions, and the size of a few units.

Run:  python3 gd_rrxor.py              (about twenty seconds: one trajectory, four initial scales, seven seeds)
      python3 gd_rrxor.py 0.03         (the trajectory for one initial scale)
"""
import itertools
import math
import random
import sys

LN2 = math.log(2)
L, D, M = 12, 4, 6          # tokens, attention offsets 0..D-1, neurons
LR = 0.5


def make_cells():
    cells = {}
    free = [p for p in range(1, L + 1) if p % 3]
    for bits in itertools.product((0, 1), repeat=len(free)):
        seq, it = [], iter(bits)
        for p in range(1, L + 1):
            seq.append(seq[-1] ^ seq[-2] if p % 3 == 0 else next(it))
        w = 1.0 / (2 ** len(free) * (L - 1))
        for t in range(1, L):
            key = (t,) + tuple((2 * seq[t - 1 - k] - 1) if t - 1 - k >= 0 else 0 for k in range(D))
            c = cells.setdefault(key, [0.0, 0.0])
            c[0] += w
            c[1] += w * seq[t]
    out = []
    for (t, *xs), (w, wy) in cells.items():
        out.append((t, tuple(xs), tuple(d for d in range(D) if t - d >= 1), w, wy / w))
    return out


CELLS = make_cells()
XPOS = {t for t in range(1, L) if (t + 1) % 3 == 0}   # positions whose next token is an XOR


class Net:
    def __init__(self, eps, seed):
        rnd = random.Random(seed)
        g = lambda: rnd.gauss(0, eps)
        self.b = [g() for _ in range(D)]
        self.beta = g()
        self.omega = g()
        self.u = [g() for _ in range(M)]
        self.v = [g() for _ in range(M)]
        self.c = [g() for _ in range(M)]
        self.gate = [[g() for _ in range(L)] for _ in range(M)]
        self.r = [g() for _ in range(M)]
        self.rho_x, self.rho_A, self.bias = g(), g(), g()
        self.d = [g() for _ in range(L)]

    def step(self, lr):
        """One full-batch gradient step. Returns loss (bits) and diagnostics computed before the step."""
        b, beta, omega, u, v, c, gate, r = self.b, self.beta, self.omega, self.u, self.v, self.c, self.gate, self.r
        gb = [0.0] * D
        gbeta = gomega = grx = grA = gbias = 0.0
        gu, gv, gc, gr = [0.0] * M, [0.0] * M, [0.0] * M, [0.0] * M
        ggate = [[0.0] * L for _ in range(M)]
        gd = [0.0] * L
        loss = confX = confR = wX = wR = att1 = watt = 0.0
        for t, xs, ds, w, ybar in CELLS:
            x0 = xs[0]
            sc = [b[d] + beta * x0 * xs[d] for d in ds]
            mx = max(sc)
            ex = [math.exp(s - mx) for s in sc]
            zs = sum(ex)
            al = [e / zs for e in ex]
            a = sum(al[i] * xs[d] for i, d in enumerate(ds))
            A = omega * a
            pre = [u[j] * x0 + v[j] * A + gate[j][t] + c[j] for j in range(M)]
            z = self.rho_x * x0 + self.rho_A * A + self.d[t] + self.bias
            for j in range(M):
                if pre[j] > 0:
                    z += r[j] * pre[j]
            p = 1 / (1 + math.exp(-z)) if z > -30 else math.exp(z)
            loss += w * ((max(z, 0) + math.log1p(math.exp(-abs(z)))) - ybar * z)
            if t >= 2:
                conf = p if x0 != xs[1] else 1 - p     # probability given to "next = XOR of the last two"
                if t in XPOS:
                    confX += w * conf
                    wX += w
                else:
                    confR += w * conf
                    wR += w
            if len(ds) == D:
                att1 += w * al[1]
                watt += w
            dz = w * (p - ybar)
            grx += dz * x0
            grA += dz * A
            gbias += dz
            gd[t] += dz
            dA = dz * self.rho_A
            for j in range(M):
                if pre[j] > 0:
                    gr[j] += dz * pre[j]
                    dp = dz * r[j]
                    gu[j] += dp * x0
                    gv[j] += dp * A
                    gc[j] += dp
                    ggate[j][t] += dp
                    dA += dp * v[j]
            gomega += dA * a
            da = dA * omega
            for i, d in enumerate(ds):
                dsd = al[i] * da * (xs[d] - a)
                gb[d] += dsd
                gbeta += dsd * x0 * xs[d]
        for d in range(D):
            b[d] -= lr * gb[d]
        self.beta -= lr * gbeta
        self.omega -= lr * gomega
        self.rho_x -= lr * grx
        self.rho_A -= lr * grA
        self.bias -= lr * gbias
        for t in range(L):
            self.d[t] -= lr * gd[t]
        for j in range(M):
            u[j] -= lr * gu[j]
            v[j] -= lr * gv[j]
            c[j] -= lr * gc[j]
            r[j] -= lr * gr[j]
            for t in range(L):
                gate[j][t] -= lr * ggate[j][t]
        return loss / LN2, confX / wX, confR / wR, att1 / watt

    def sizes(self):
        rs = sum(abs(x) for x in self.r) / M
        tot = sum(abs(x) for x in self.r) or 1.0
        contrast = sum(abs(self.r[j]) * (sum(self.gate[j][t] for t in XPOS) / len(XPOS)
                                         - sum(self.gate[j][t] for t in range(2, L) if t not in XPOS) / (L - 2 - len(XPOS)))
                       for j in range(M)) / tot
        return abs(self.omega), rs, contrast


def run(eps, seed=0, max_steps=40000, verbose=False, stop=0.645):
    """Train one network. Returns the times at which the loss first falls below 0.95 (soft XOR) and 0.80
    (gating), the moment of greatest confidence at non-XOR positions, and samples of the leading
    three-weight product |r_j * v_j * omega|."""
    net = Net(eps, seed)
    if verbose:
        print(f"{'time':>8s} {'loss':>7s} {'P(xor) at XOR pos':>18s} {'at other pos':>13s} {'attn 1 back':>12s} {'|omega|':>8s} {'mean|r|':>8s} {'gate contrast':>14s}")
    out = {'soft': None, 'gate': None, 'peak': (0.0, 0.0, 0.0, 0.0), 'product': []}
    next_print = 0
    for s in range(max_steps + 1):
        loss, cx, cr, a1 = net.step(LR)
        time = s * LR
        if out['soft'] is None and loss < 0.95:
            out['soft'] = time
        if out['gate'] is None and loss < 0.80:
            out['gate'] = time
        if cr > out['peak'][0]:
            out['peak'] = (cr, cx, loss, time)
        if s % 250 == 0:
            out['product'].append((time, max(abs(net.r[j] * net.v[j] * net.omega) for j in range(M)) ** (1 / 3)))
        if verbose and s >= next_print:
            om, rs, con = net.sizes()
            print(f"{time:8.0f} {loss:7.4f} {cx:18.3f} {cr:13.3f} {a1:12.3f} {om:8.3f} {rs:8.3f} {con:14.3f}")
            next_print = max(s + 1, int(s * 1.45))
        if loss < stop:
            break
    out['final'] = loss
    return out


def main():
    if len(sys.argv) > 1:
        for e in (float(a) for a in sys.argv[1:]):
            print(f"\n=== initial scale {e} ===")
            run(e, verbose=True)
        return
    print("=== one run, initial scale 0.1 ===")
    run(0.1, verbose=True)

    print("\n=== initial scale: how long is the plateau? ===")
    print(f"{'scale':>6s} {'soft XOR at':>12s} {'scale x time':>13s} {'then gating after':>18s}   soft stage: P(xor) at XOR positions / elsewhere, loss")
    traces = {}
    for e in (0.3, 0.1, 0.03, 0.01):
        o = traces[e] = run(e)
        cr, cx, ls, _ = o['peak']
        print(f"{e:6.2f} {o['soft']:12.0f} {e * o['soft']:13.1f} {o['gate'] - o['soft']:18.0f}   {cx:.3f} / {cr:.3f}, {ls:.4f}")

    print("\n=== is the plateau a product of three weights? (initial scale 0.03) ===")
    print("   s = (|r_j v_j omega|)^(1/3) for the leading neuron; ds/dt proportional to s^2 means 1/s falls linearly")
    for time, s in traces[0.03]['product']:
        if time <= traces[0.03]['soft']:
            print(f"   time {time:6.0f}   1/s = {1 / s:6.2f}")

    print("\n=== seeds at initial scale 0.1 ===")
    softs, gaps = [], []
    for seed in range(7):
        o = run(0.1, seed=seed, stop=0.70)
        softs.append(o['soft'])
        gaps.append(o['gate'] - o['soft'])
        print(f"   seed {seed}: soft XOR at {o['soft']:6.0f}, gating {o['gate'] - o['soft']:4.0f} later; soft stage P(xor) {o['peak'][1]:.3f} / {o['peak'][0]:.3f}")
    for name, xs in (("time to soft XOR", softs), ("soft XOR to gating", gaps)):
        mean = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
        print(f"   {name}: mean {mean:.0f}, standard deviation / mean = {sd / mean:.2f}")


if __name__ == '__main__':
    main()
