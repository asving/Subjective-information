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

Second use: the phase circuit of random-start RRXOR, with units written by hand above the level of
neurons. Two primitives: the XOR of two columns (with tokens +-1, their product), and an OR pooled over
every third position back. The shapes come from rrxor_random.py.

    xor         e_t = x_t * x_(t-1)                                   factors (r_e, a)
    two back    x_(t-2), a cheap trick                                factors (r_s, s)
    cancel r    e_t * D_r, for r = 2, 0, 1, where v_t = x_t * e_(t-1) marks a token that breaks the XOR rule
                and D_r = OR of v at r, r+3, ... positions back       factors (rho_r, f_r, c_r, b, a, a)
    sure        e_t * D_0 * D_1                                       factors (rho_s, f_s, c_0, c_1, b, b, a, a, a)

Each use of a primitive has one write weight (a for e, b for v, c_r for D_r, f for a product) and the output
reads each product with one weight. That count is a choice, not derived from a network. All factors start
as Gaussians of scale x0. The race is run as gradient flow and under Adam with exact gradients.

Run:  python3 unit_race.py            (sticky RRXOR: the table; about a minute)
      python3 unit_race.py trace      (sticky RRXOR, one run in detail: s = 0.25, initial scale 0.01, seed 0)
      python3 unit_race.py phase      (the phase circuit; a few seconds)
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

    def adam_step(self, lr=0.01):
        """One Adam step on the exact gradient. Returns the loss and probabilities before the step."""
        if not hasattr(self, 'm'):
            self.m, self.v, self.n = dict.fromkeys(self.w, 0.0), dict.fromkeys(self.w, 0.0), 0
        loss, p, rate = self.rates(self.w)
        self.n += 1
        for f in self.w:
            g = -rate[f]
            self.m[f] = 0.9 * self.m[f] + 0.1 * g
            self.v[f] = 0.999 * self.v[f] + 0.001 * g * g
            self.w[f] -= lr * (self.m[f] / (1 - 0.9 ** self.n)) / (math.sqrt(self.v[f] / (1 - 0.999 ** self.n)) + 1e-8)
        self.t += 1
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


# ---------------------------------------------------------------- the phase circuit of random-start RRXOR

def phase_menu():
    """Hand-written units for random-start RRXOR, built from two primitives: the XOR of two columns (a product,
    with tokens +-1) and an OR pooled over every third position back. Returns (cells, units)."""
    import rrxor_random as rr
    X = rr.X
    XOR = ('head', ('off', 1), X, X, 'mean')            # e_t = x_t * x_(t-1)
    VIO = ('head', ('off', 1), X, XOR, 'mean')          # v_t = x_t * e_(t-1): +1 where token t breaks the XOR rule
    e = rr.val(XOR)
    D = {r: [1.0 if m == 1 else 0.0 for m in rr.val(('head', ('per', 3, r), None, VIO, 'max'))] for r in (2, 0, 1)}
    shapes = [('xor', ('r_e', 'a'), e),
              ('two back', ('r_s', 's'), rr.val(('head', ('off', 2), None, X, 'mean')))]
    for r in (2, 0, 1):
        shapes.append((f'cancel {r}', (f'rho{r}', f'f{r}', f'c{r}', 'b', 'a', 'a'), [x * d for x, d in zip(e, D[r])]))
    shapes.append(('sure', ('rho_s', 'f_s', 'c0', 'c1', 'b', 'b', 'a', 'a', 'a'), [x * d0 * d1 for x, d0, d1 in zip(e, D[0], D[1])]))
    agg = {}
    for key, w, y in zip(zip(*[sh[2] for sh in shapes]), rr.W, rr.Y):
        c = agg.setdefault(key, [0.0, 0.0])
        c[0] += w
        c[1] += y
    keys = list(agg)
    cells = [(agg[k][0], agg[k][1] / agg[k][0]) for k in keys]
    units = [(name, F, [(i, k[j]) for i, k in enumerate(keys) if k[j]]) for j, (name, F, _) in enumerate(shapes)]
    return cells, units


def best_readout(cells, units):
    """Loss of the best linear readout of the units' shapes: the same race with one weight per unit."""
    race = Race(cells, [(n, (n,), phi) for n, _, phi in units], {n: 0.0 for n, _, _ in units})
    for _ in range(4000):
        loss, _ = race.step(tol=0.05, dt_max=20.0)
    return loss


def run_phase(cells, units, x0, seed, adam, ref, cap=12000):
    """One race. Returns [(time, loss, strengths)], strength = product of a unit's factors."""
    import random
    rnd = random.Random(seed)
    names = sorted({f for _, F, _ in units for f in F})
    race = Race(cells, units, {f: rnd.gauss(0, x0) for f in names}, floor=x0 / 10)
    hist = []
    while len(hist) < cap:
        st = []
        for _, F, _ in units:
            x = 1.0
            for f in F:
                x *= race.w[f]
            st.append(x)
        t = race.t
        loss, _ = race.adam_step(0.01) if adam else race.step()
        hist.append((t, loss, st))
        if loss < ref + 0.0005:
            break
    return hist


def part_phase():
    cells, units = phase_menu()
    ref = best_readout(cells, units)
    print("=== the phase circuit of random-start RRXOR, from two primitives, raced in time ===")
    print(f"    {len(cells)} cells, {len(units)} units; best linear readout of their shapes: {ref:.4f} bits per token")
    print("    each unit: time (flow) or step (Adam) at which its strength first reaches half its final value; loss at that moment")
    for adam, x0 in ((False, 0.1), (False, 0.5), (True, 0.1), (True, 0.5)):
        for seed in range(3):
            hist = run_phase(cells, units, x0, seed, adam, ref)
            final = hist[-1][2]
            arrive = []
            for j, (name, _, _) in enumerate(units):
                if abs(final[j]) < 0.02:
                    continue
                t, loss = next((t, loss) for t, loss, st in hist if st[j] * final[j] > 0 and abs(st[j]) >= 0.5 * abs(final[j]))
                arrive.append((t, name, loss, final[j]))
            arrive.sort()
            j = [u[0] for u in units].index('two back')
            pk = max(hist, key=lambda row: abs(row[2][j]))
            print(f"  {'Adam' if adam else 'flow'}, x0 = {x0}, seed {seed}: ends at {hist[-1][0]:.0f} with loss {hist[-1][1]:.4f}")
            print("      " + ';  '.join(f"{name} at {t:.0f} (loss {loss:.4f}, strength {fs:+.2f})" for t, name, loss, fs in arrive))
            half = next(t for t, _, st in hist if st[j] * pk[2][j] > 0 and abs(st[j]) >= 0.5 * abs(pk[2][j]))
            print(f"      two back: half of its largest strength at {half:.0f}, largest {pk[2][j]:+.2f} at {pk[0]:.0f}, final {final[j]:+.2f}", flush=True)
    for adam, seed in ((False, 1), (True, 1)):
        hist = run_phase(cells, units, 0.1, seed, adam, ref)
        print(f"\n  one run in detail: {'Adam' if adam else 'gradient flow'}, x0 = 0.1, seed {seed}; strength of each unit")
        print(f"  {'time' if not adam else 'step':>6s} {'loss':>7s}  " + ' '.join(f"{u[0]:>9s}" for u in units))
        last = -1.0
        for t, loss, st in hist:
            if t >= last * (1.6 if not adam else 1.25) + (5 if not adam else 8) or t == hist[-1][0]:
                print(f"  {t:6.0f} {loss:7.4f}  " + ' '.join(f"{x:+9.3f}" for x in st))
                last = t


if __name__ == '__main__':
    if sys.argv[1:] == ['trace']:
        main_trace()
    elif sys.argv[1:] == ['phase']:
        part_phase()
    else:
        main()
