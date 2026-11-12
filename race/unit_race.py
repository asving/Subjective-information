"""A race between hand-written circuits, run in time.

The model. A unit is a set of named weights, its factors, and an output shape phi: one number per input.
The prediction on an input is

    z = sum over units u of ( product of the factors of u ) * phi_u(input)

and the factors follow gradient flow on the exact log loss. Units may share factors. Race below is that
and nothing more; it knows nothing about transformers. A circuit enters as a list of units, and its
implementation in a network decides which weights are its factors and at what size they start.

First test: sticky RRXOR in the toy transformer of gd_rrxor.py. The data are RRXOR with an aligned start,
except that the first coin of each block repeats the first coin of the previous block with probability
1/2 + s. The network then hosts two circuits that share one weight, the head's write omega:

    copy          the token two back, through uniform attention and the unembedding's read of the head.
                  Factors {rho_A, omega}; shape = mean of the attended tokens.
    soft XOR j    through neuron j, as in unit_price.py. Factors {omega, v_j, r_j};
                  shape = [neuron j on at the start] * mean of the attended tokens.

Three simulators are started from the literal network's own initial weights and compared with it:

    two       only the two kinds of unit above: 7 units, 14 factors
    frozen    every first-order unit of the network (each read and write of each neuron, the direct paths,
              the biases); the neurons' on-patterns stay as they were at the start; attention stays uniform
    live      the same, with the on-patterns recomputed from the current weights. This is the literal
              network with its attention pattern held uniform

Three events are timed: |omega| first above 0.3 (the shared weight is on), and the probability given to the
XOR at XOR positions first above 0.52 (the XOR appears) and first above 0.6 (the XOR is strong).

Run:  python3 unit_race.py            (the table; about a minute)
      python3 unit_race.py trace      (one run in detail: s = 0.25, initial scale 0.01, seed 0)
"""
import itertools
import math
import sys

import gd_rrxor as gd
from gd_rrxor import Net, L, D, M

LN2 = math.log(2)


# ---------------------------------------------------------------- the race

class Race:
    """Gradient flow on the factors of a list of units.

    cells     list of (weight, probability that the outcome is 1), one per input
    units     list of (name, factors, phi); phi is a list of (cell index, value) for the nonzero values
    w         dict: factor -> initial value
    reshape   optional function w -> units, for output shapes that depend on the current weights
    """

    def __init__(self, cells, units, w, reshape=None, floor=0.01):
        self.cw = [c[0] for c in cells]
        self.cy = [c[1] for c in cells]
        self.units, self.w, self.reshape, self.floor = units, dict(w), reshape, floor
        self.t = 0.0

    def rates(self, w):
        """Loss in bits, predicted probabilities, and dw/dt for every factor."""
        units = self.reshape(w) if self.reshape else self.units
        z = [0.0] * len(self.cw)
        strength = []
        for _, F, phi in units:
            s = 1.0
            for f in F:
                s *= w[f]
            strength.append(s)
            if s:
                for c, v in phi:
                    z[c] += s * v
        loss, p, delta = 0.0, [], []
        for zc, wc, yc in zip(z, self.cw, self.cy):
            pc = 1 / (1 + math.exp(-zc)) if zc > -30 else math.exp(zc)
            p.append(pc)
            delta.append(wc * (pc - yc))
            loss += wc * (max(zc, 0) + math.log1p(math.exp(-abs(zc))) - yc * zc)
        rate = dict.fromkeys(w, 0.0)
        for _, F, phi in units:
            G = 0.0
            for c, v in phi:
                G += delta[c] * v
            if G:
                for i, f in enumerate(F):
                    o = 1.0
                    for i2, f2 in enumerate(F):
                        if i2 != i:
                            o *= w[f2]
                    rate[f] -= G * o
        return loss / LN2, p, rate

    def step(self, tol=0.02, dt_max=500.0):
        """One midpoint step, sized so that no factor changes by more than about tol of its size.
        Returns the loss and probabilities at the start of the step."""
        w = self.w
        loss, p, r1 = self.rates(w)
        speed = max(abs(r1[f]) / (abs(w[f]) + self.floor) for f in w)
        dt = min(dt_max, tol / speed) if speed > 0 else dt_max
        _, _, r2 = self.rates({f: w[f] + 0.5 * dt * r1[f] for f in w})
        self.w = {f: w[f] + dt * r2[f] for f in w}
        self.t += dt
        return loss, p


# ---------------------------------------------------------------- sticky RRXOR and the toy transformer's units

def sticky_cells(s):
    """The cells of gd_rrxor.make_cells, with the first coin of each block sticky to the previous block's."""
    cells = {}
    free = [p for p in range(1, L + 1) if p % 3]
    for bits in itertools.product((0, 1), repeat=len(free)):
        seq, it = [], iter(bits)
        for p in range(1, L + 1):
            seq.append(seq[-1] ^ seq[-2] if p % 3 == 0 else next(it))
        pr = 0.5 ** 5
        for b in range(1, L // 3):
            pr *= 0.5 + s if seq[3 * b] == seq[3 * b - 3] else 0.5 - s
        w = pr / (L - 1)
        for t in range(1, L):
            key = (t,) + tuple((2 * seq[t - 1 - k] - 1) if t - 1 - k >= 0 else 0 for k in range(D))
            c = cells.setdefault(key, [0.0, 0.0])
            c[0] += w
            c[1] += w * seq[t]
    out = []
    for (t, *xs), (w, wy) in cells.items():
        out.append((t, tuple(xs), tuple(d for d in range(D) if t - d >= 1), w, wy / w))
    return out


def weights_of(net):
    w = {'rho_x': net.rho_x, 'rho_A': net.rho_A, 'bias': net.bias, 'omega': net.omega}
    for t in range(L):
        w[f'd{t}'] = net.d[t]
    for j in range(M):
        w[f'r{j}'], w[f'u{j}'], w[f'v{j}'], w[f'c{j}'] = net.r[j], net.u[j], net.v[j], net.c[j]
        for t in range(L):
            w[f'q{j}_{t}'] = net.gate[j][t]
    return w


def menu(cells, w0, kind):
    """The units of the toy transformer to first order in its weights. Returns (units, reshape)."""
    abar = [sum(xs[d] for d in ds) / len(ds) for _, xs, ds, _, _ in cells]
    n = len(cells)

    def on(w, j, live):
        """The cells on which neuron j is on."""
        out = []
        for i, (t, xs, _, _, _) in enumerate(cells):
            pre = w[f'u{j}'] * xs[0] + w[f'q{j}_{t}'] + w[f'c{j}']
            if live:
                pre += w[f'v{j}'] * w['omega'] * abar[i]
            if pre > 0:
                out.append(i)
        return out

    def neuron_units(w, j, live, everything):
        cs = on(w, j, live)
        units = [(f'xor{j}', (f'r{j}', f'v{j}', 'omega'), [(i, abar[i]) for i in cs if abar[i]])]
        if everything:
            units.append((f'tok{j}', (f'r{j}', f'u{j}'), [(i, cells[i][1][0]) for i in cs]))
            units.append((f'const{j}', (f'r{j}', f'c{j}'), [(i, 1.0) for i in cs]))
            by_t = {}
            for i in cs:
                by_t.setdefault(cells[i][0], []).append((i, 1.0))
            for t, phi in by_t.items():
                units.append((f'pos{j}_{t}', (f'r{j}', f'q{j}_{t}'), phi))
        return units

    copy = [('copy', ('rho_A', 'omega'), [(i, abar[i]) for i in range(n) if abar[i]])]
    if kind == 'two':
        return copy + [u for j in range(M) for u in neuron_units(w0, j, False, False)], None
    fixed = copy + [('token', ('rho_x',), [(i, cells[i][1][0]) for i in range(n)]),
                    ('bias', ('bias',), [(i, 1.0) for i in range(n)])]
    for t in range(1, L):
        fixed.append((f'bias{t}', (f'd{t}',), [(i, 1.0) for i in range(n) if cells[i][0] == t]))
    if kind == 'frozen':
        return fixed + [u for j in range(M) for u in neuron_units(w0, j, False, True)], None
    return None, lambda w: fixed + [u for j in range(M) for u in neuron_units(w, j, True, True)]


def xor_confidence(cells, p):
    """Probability given to 'next = XOR of the last two tokens', averaged over XOR positions."""
    num = den = 0.0
    for (t, xs, _, w, _), pc in zip(cells, p):
        if t in gd.XPOS:
            num += w * (pc if xs[0] != xs[1] else 1 - pc)
            den += w
    return num / den


OMEGA_ON, XOR_ONSET, XOR_STRONG = 0.3, 0.52, 0.6
EVENTS = ('omega', 'onset', 'strong')


def note(ev, t, om, cx):
    if ev['omega'] is None and om >= OMEGA_ON:
        ev['omega'] = t
    if ev['onset'] is None and cx >= XOR_ONSET:
        ev['onset'] = t
    if ev['strong'] is None and cx >= XOR_STRONG:
        ev['strong'] = t


def simulate(cells, net, kind, t_max=60000.0, trace=None):
    """Times of the three events in one simulator. trace: optional list collecting (time, loss, xor, omega)."""
    w0 = weights_of(net)
    if kind == 'two':
        keep = {'rho_A', 'omega'} | {f'{a}{j}' for a in 'rv' for j in range(M)}
        w0 = {f: x for f, x in w0.items() if f in keep}
    units, reshape = menu(cells, weights_of(net), kind)
    race = Race([(c[3], c[4]) for c in cells], units, w0, reshape)
    ev = dict.fromkeys(EVENTS)
    while race.t < t_max and ev['strong'] is None:
        t = race.t
        om = abs(race.w['omega'])
        loss, p = race.step()
        cx = xor_confidence(cells, p)
        note(ev, t, om, cx)
        if trace is not None:
            trace.append((t, loss, cx, om))
    return ev


def literal(cells, net, t_max=60000.0, trace=None):
    """The same three events in the network itself, under gradient descent with gd_rrxor's learning rate."""
    gd.CELLS = cells
    ev = dict.fromkeys(EVENTS)
    s = 0
    while s * gd.LR < t_max and ev['strong'] is None:
        om = abs(net.omega)
        loss, cx, _, a1 = net.step(gd.LR)
        if ev['onset'] is None and cx >= XOR_ONSET:
            ev['attention'] = a1
        note(ev, s * gd.LR, om, cx)
        if trace is not None and s % 20 == 0:
            trace.append((s * gd.LR, loss, cx, om))
        s += 1
    return ev


def copy_gain(cells):
    return sum(w * (y - 0.5) * sum(xs[d] for d in ds) / len(ds) for _, xs, ds, w, y in cells)


def show(x):
    return f"{x:6.0f}" if x is not None else "  none"


def ratio(a, b):
    return f"{a / b:5.2f}" if a is not None and b else " none"


def main():
    kinds = ('two', 'frozen', 'live')
    print("=== sticky RRXOR in the toy transformer: the literal network against three simulators ===")
    print("    literal: time at which |omega| passes 0.3 / the XOR appears (0.52) / the XOR is strong (0.6)")
    print("    each simulator: its three times divided by the literal ones (none = never within 4 times the literal run)")
    print(f"{'scale':>6s} {'s':>5s} {'seed':>4s} {'copy gain':>10s}   {'literal':>20s}   " + '   '.join(f"{k:>17s}" for k in kinds))
    summary = {}
    for eps in (0.1, 0.01):
        for s in (0.0, 0.1, 0.25, 0.4):
            cells = sticky_cells(s)
            for seed in range(3):
                lit = literal(cells, Net(eps, seed))
                row = f"{eps:6.2f} {s:5.2f} {seed:4d} {copy_gain(cells):10.4f}   " + '/'.join(show(lit[e]) for e in EVENTS)
                for k in kinds:
                    sim = simulate(cells, Net(eps, seed), k, t_max=4 * (lit['strong'] or 60000.0))
                    row += "   " + ' '.join(ratio(sim[e], lit[e]) for e in EVENTS)
                    for e in EVENTS:
                        summary.setdefault((eps, s, k, e), []).append(sim[e] / lit[e] if sim[e] is not None and lit[e] else None)
                summary.setdefault((eps, s), []).append(lit)
                print(row, flush=True)
    print("\n    simulator / literal, range over the three seeds")
    for eps in (0.1, 0.01):
        for s in (0.0, 0.1, 0.25, 0.4):
            for k in kinds:
                parts = []
                for e in EVENTS:
                    xs = summary[(eps, s, k, e)]
                    got = [x for x in xs if x is not None]
                    parts.append(f"{e} {min(got):.2f} to {max(got):.2f}" + (f" ({len(xs) - len(got)} of {len(xs)} never)" if len(got) < len(xs) else "")
                                 if got else f"{e} never")
                print(f"    scale {eps}, s = {s:4.2f}, {k:>6s}: " + ';  '.join(parts))
    print("\n    how much sooner the XOR appears in the literal network than at s = 0, same seed")
    for eps in (0.1, 0.01):
        for s in (0.1, 0.25, 0.4):
            print(f"    scale {eps}, s = {s}: " + ', '.join(f"{a['onset'] / b['onset']:.1f}x" for a, b in zip(summary[(eps, 0.0)], summary[(eps, s)])))
    print("\n    the head's attention to the token one back when the XOR appears in the literal network (uniform: 0.25)")
    for eps in (0.1, 0.01):
        for s in (0.0, 0.1, 0.25, 0.4):
            print(f"    scale {eps}, s = {s}: " + ', '.join(f"{lit['attention']:.3f}" for lit in summary[(eps, s)]))


def main_trace(s=0.25, eps=0.01, seed=0):
    cells = sticky_cells(s)
    print(f"=== one run: s = {s}, initial scale {eps}, seed {seed}; copy gain {copy_gain(cells):.4f} ===")
    for name in ('literal', 'two', 'frozen', 'live'):
        tr = []
        if name == 'literal':
            ev = literal(cells, Net(eps, seed), trace=tr)
            t_max = 4 * ev['strong']
        else:
            ev = simulate(cells, Net(eps, seed), name, t_max=t_max, trace=tr)
        print(f"  {name}: |omega| on at {show(ev['omega'])}, XOR appears at {show(ev['onset'])}, strong at {show(ev['strong'])}")
        print(f"    {'time':>8s} {'loss':>8s} {'P(xor)':>7s} {'|omega|':>8s}")
        for row in tr[::max(1, len(tr) // 14)] + tr[-1:]:
            print(f"    {row[0]:8.0f} {row[1]:8.5f} {row[2]:7.3f} {row[3]:8.4f}")


if __name__ == '__main__':
    if sys.argv[1:] == ['trace']:
        main_trace()
    else:
        main()
