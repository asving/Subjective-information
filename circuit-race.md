# A race between circuits

*Which circuit does a network learn first, and when? Notes from a discussion between Asvin G. and Claude (Fable 5.2), November 2026. Every number below is printed by a script in `race/`.*

## The claim

A network's circuits can be described as *units* that read from and write to a shared stream. When a unit appears is set by three things: how many of its weights still have to grow, how strongly its output correlates with the current error, and the optimiser. Under plain gradient descent the wait multiplies with every extra weight. Under Adam it barely depends on them. That holds when gradients are exact or batches are large. With small batches of fresh samples in high dimension the wait grows steeply for both, and Adam is ahead but no longer flat.

## The model

**Units on a stream.** The stream is the residual stream: one column per feature, one row per position. A unit reads some columns and writes one. An attention head and an MLP neuron are both units, and so is any hand-written piece of circuit, such as "multiply the current token by the previous one". A unit that reads another unit's column depends on it; these dependencies form a graph without cycles.

**A unit is a path of weights that multiply.** A head's write, the read of the neuron that takes it, that neuron's write: the unit's output is proportional to their product. Each unit on a path contributes a read and a write. The weights on the path that are still at their small initial size x₀ are its *dormant factors*; say there are k. A unit all of whose factors have grown is *active*.

**Gain.** The gain g of a dormant unit is the correlation of its output, at unit size, with the current prediction error. For a single neuron this is the utility of Alternating Gradient Flows (Kunin, Marchetti et al., 2025), whose alternation this model follows: dormant units accumulate gain until one activates, then the active units refit and the error changes.

**The wait.** Under plain gradient descent each dormant factor grows at g times the product of the others, so the unit activates after

    h_k(x₀) / g       h_1 = 1 − x₀,    h_2 = log(1/x₀),    h_k = (x₀^(2−k) − 1) / (k − 2)  for k ≥ 3

Four consequences follow.

- **Depth multiplies.** Each dormant factor beyond the second multiplies the wait by about 1/x₀.
- **Bundles.** A unit that is useless alone, such as a head that only copies, is paid for inside the bundle that uses it, as extra factors.
- **Reuse is cheap.** A unit that reads active units pays only for its own factors. A new read on an active neuron has no wait at all.
- **Pull-back.** Output that is not part of the target is error, so the loss pulls back the weights that produce it. A dormant unit that already emits something feels this as a drag.

**Adam.** Adam moves each weight by about one learning rate per step, whatever the size of its gradient. With exact gradients the wait becomes a distance to travel and no longer depends on k or x₀. With noisy gradients it does not (section 3).

## Evidence

### 1. A chain of weights

A product of k weights, each starting at 0.1, fitted to a constant. Time until the product passes half its target (`unit_price.py chain`):

| k | h_k(0.1) | Plain gradient descent | Adam |
|---|---|---|---|
| 2 | 2.3 | 2.3 | 0.54 |
| 3 | 9.0 | 9.2 | 0.56 |
| 4 | 49.5 | 49.8 | 0.58 |
| 6 | 2,499.7 | 2,500.1 | 0.61 |

### 2. A one-layer transformer on RRXOR

The data are twelve tokens R R X R R X …, with R a fair coin and X the XOR of the two tokens before it. `gd_rrxor.py` trains a one-layer transformer (one head, six ReLU neurons, every weight a scalar) by full-batch gradient descent from weights of scale ε.

Its first circuit is a bundle. The head copies earlier tokens, which is useless alone, and a neuron reads the copy together with the current token. There are three dormant factors: the head's write, the neuron's read and the neuron's write. The network then passes through two stages, which the model reads as reuse.

- **Plateau.** Integrating the three-factor equations from the network's own initial weights predicts when the plateau ends, with nothing fitted (`unit_price.py plateau`).

| Initial scale ε | Actual end of plateau, 7 seeds | Predicted / actual |
|---|---|---|
| 0.01 | 2,159 to 13,144 | 0.97 to 1.02 |
| 0.1 | 244 to 1,408 | 0.74 to 1.33 |

- **Soft rule first.** The circuit that appears predicts the XOR at every position, with probability about 0.72 at XOR positions and 0.68 elsewhere. It is wrong half the time at non-XOR positions and still lowers the loss.
- **Then a gate.** The neuron gains a read of position, one new factor on an active unit, and switches itself off at non-XOR positions. This follows 220 to 366 time units later at every ε, while the plateau grows as 1/ε.
- **Under Adam** (`unit_price.py adam`) the plateau ends at step 54 to 56 for ε from 0.3 down to 0.01. Seven seeds give 46 to 68 steps, apart from one that takes over a thousand.

### 3. The leap staircase

Abbe, Boix-Adserà and Misiakiewicz (2023) train a two-layer network on targets that are sums of monomials in a few coordinates of x ∈ {+1, −1}^d. They find that a monomial bringing in L new coordinates takes about d^(L−1) steps of online SGD, and that a staircase such as z₁ + z₁z₂ + z₁z₂z₃ is learnt quickly because each step brings in one coordinate.

In the model a unit is a pair (neuron, monomial). Its dormant factors are the neuron's output weight and its first-layer weights on the new coordinates, L + 1 in all, and its gain is the monomial's coefficient times the mean L-th derivative of the activation. `leap.py` checks this with exact gradients (all 2^d inputs).

- **Plateau, plain gradient descent.** The bare factor equations run early, because a sigmoid neuron emits output before it has learnt anything and the loss pulls its weights back. With that pull-back included, and nothing fitted:

| Neurons, d | L | Actual end of plateau, 3 seeds | Bare / actual | With pull-back / actual |
|---|---|---|---|---|
| 16, 10 | 2 | 189 to 283 | 0.76 to 0.86 | 1.02 to 1.05 |
| 16, 10 | 3 | 560 to 1,979 | 0.59 to 0.79 | 0.96 to 1.00 |
| 16, 10 | 4 | 11,389 to 16,901 | 0.27 to 0.60 | 0.93 to 1.02 |
| 64, 8 | 2 | 89 to 154 | 0.86 to 0.91 | 1.00 to 1.02 |
| 64, 8 | 3 | 376 to 671 | 0.80 to 0.86 | 0.94 to 0.96 |
| 64, 8 | 4 | 2,957 to 5,286 | 0.64 to 0.69 | 0.86 to 0.91 |

- **Staircase against monomial.** Under plain gradient descent the staircase reaches z₁z₂z₃ at time 56 to 66. The monomial z₁z₂z₃ alone takes 558 to 854. This is reuse: each step of the staircase adds one factor to neurons that are already active.
- **Adam, exact gradients.** The plateau lasts 29 to 82 steps for L = 2, 3, 4 at initial scales 0.3 and 0.1. At L = 4 and scales 0.03 and 0.01 it lasts 146 to 287 steps, so the price is flat only approximately. The staircase gives Adam no advantage: both targets take about 80 to 120 steps.
- **Fresh samples every step** (`leap.py online`: 64 neurons, batch 32, median steps over three seeds until the test loss is 20% under that of predicting zero).

| Target | Optimiser | d = 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| z₁z₂ | SGD | 600 | 1,400 | 2,100 | 3,600 | 9,300 | |
| z₁z₂ | Adam | 400 | 500 | 600 | 900 | 1,500 | |
| z₁z₂z₃ | SGD | 1,400 | 3,000 | 6,700 | 178,700 | 1,023,900 | |
| z₁z₂z₃ | Adam | 500 | 700 | 1,300 | 3,300 | 120,700 | about 2,700,000 |

At leap 2, SGD's steps grow as d^1.07 from d = 64 to 256, which is the paper's d^(L−1), and Adam's as d^0.66. At leap 3 both rise far more steeply than d² at these sizes: SGD between d = 64 and 128, Adam one doubling later. At d = 512 one Adam seed of three did not finish within 3,000,000 steps, and at d = 1,024 none did. Larger batches shorten the wait for both. At d = 128 and leap 3, Adam takes 10,600, 3,300, 1,100 and 500 steps at batch 8, 32, 128 and 512, and SGD takes 276,500, 178,700, 22,200 and 26,500.

Our reading: with fresh samples the gradient of a dormant unit is a small signal, a product of L small weights, under noise that does not shrink with it. Adam can rescale a small gradient but cannot remove its noise. This was not measured directly.

The pull-back in the plateau table was found after the bare equations failed. It has no free parameter, but it was not predicted.

### 4. A race with no prices

If the wait does not depend on the number of factors, candidate units are ordered by the signal-to-noise ratio of their gradient: the gain divided by its spread per token. This ranking has no free parameter. `unit_price.py order` applies it to RRXOR started at a random phase, where the optimal predictor has to infer the phase.

- **Stage 1.** The XOR head (current token times previous token) ranks first of 3,738 candidate units and linked pairs.
- **Stage 2.** A bundle ranks first of 17,415: a slot marking where the XOR rule was violated, plus the XOR multiplied by an OR of violations every third position back. This is the circuit reported for trained transformers (A.G.), and a lookup on its slots scores 0.8245 bits per token against an optimum of 0.8234.
- **The multiplying law disagrees.** By a rough count the bundle has ten factors, and a cheap alternative (the token two back) has four. Under plain gradient descent the bundle would come first only if weights started above 0.81 of their full size.

## What the race builds

Before prices were derived, the race was run on five processes with hand-set prices of about 3 bits per column a unit names, adopting at each stage the unit or linked pair with the smallest price per bit saved. Under that rule the price grows only linearly with the columns named, far closer to a flat price than to the multiplying law. The circuits it builds match what is reported for trained transformers.

| Process | Circuit the race builds | Reported for trained transformers (A.G.) | Script |
|---|---|---|---|
| RRXOR, aligned start | Soft XOR at every position (probability 0.70), then position gates | Soft XOR, then gating (section 2 above) | `rrxor_aligned.py` |
| RRXOR, random start | XOR, then an OR over the history of XOR violations | XOR early, then that phase circuit | `rrxor_random.py` |
| Orbit v, Uv, U²v, … of an orthogonal matrix | Previous-token head plus similarity head: loss 0.495, optimum 0.270 | Fit the current token on earlier ones and shift | `orbit.py` |
| Mess3 | Token tally with strengths 1.05, 0.54, 0.31 on the last three tokens; belief readable with R² 0.985 | Belief state linearly readable (Shai et al.) | `mess3.py` |
| Self-inoculation | Swap applied everywhere at half strength, then a switch on the situation belief (flip 0.94 in A, 0.06 in B), and the blind swap rolled back from 0.53 to 0.02 | Detour through a shared flip, then a switch | `self_inoculation.py` |

Two patterns recur.

- **Soft rule, then gate, then rollback.** A cheap rule is applied everywhere at partial strength, a gate arrives, and the ungated strength returns to zero.
- **Predict, compare, pool, gate.** The circuits apply a rule to the past, compare it with what happened, pool the comparison over the history, and use the result to weight the rule now.

Only the soft XOR and the orbit circuit were computed before the trained result was known.

## What is not established

- **Units are written by hand.** The model says when a given unit appears. It does not find the units.
- **No real transformer has been compared.** The networks here are a one-layer toy with scalar weights and a two-layer network. Layer norm and weight decay are untested.
- **The five examples have not been rerun with derived prices.** Factor counts for hand-written units are rough, since a wiring does not say how many weights implement it.
- **Adam's flat price is approximate.** It rises at leap 4 from small weights, and one toy-transformer seed in seven was slow. The cause of the rise is not identified; it is not Adam's small-gradient floor.
- **The noisy regime is measured, not modelled.** With fresh samples the wait at leap 3 rises more steeply than the paper's d² at the sizes tried, with three seeds and one batch size per dimension. The model has no law for it.

## Reproducing

Plain Python 3 unless noted. Each script prints its own setup and results.

    cd race
    python3 unit_price.py          # about 1 min   sections 1, 2 and 4
    python3 gd_rrxor.py            # about 20 s    the toy transformer of section 2
    python leap.py                 # about 1 min   section 3, exact gradients (needs numpy)
    python leap.py online          # about 1.5 h  section 3, fresh samples (needs numpy)
    python3 rrxor_aligned.py       # about 1 s     the five hand-priced races
    python3 rrxor_random.py        # about 35 s
    python3 orbit.py               # about 10 s
    python3 mess3.py               # about 15 s
    python3 self_inoculation.py    # about 15 s

`torch_read_write.py` is an earlier measurement that this note no longer uses.

## References

- E. Abbe, E. Boix-Adserà, T. Misiakiewicz, *SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics*, arXiv:2302.11055, 2023.
- D. Kunin, G. L. Marchetti, F. Chen, D. Karkada, J. B. Simon, M. R. DeWeese, S. Ganguli, N. Miolane, *Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks*, NeurIPS 2025.
- G. Weiss, Y. Goldberg, E. Yahav, *Thinking Like Transformers* (RASP), ICML 2021.
- N. Elhage et al., *A Mathematical Framework for Transformer Circuits*, 2021.
- A. Shai, S. Marzen, L. Teixeira, A. Gietelink Oldenziel, P. Riechers, *Transformers represent belief state geometry in their residual stream*, arXiv:2405.15943.
- Asvin G., *Self-Inoculation*, asving.com, September 2026.
