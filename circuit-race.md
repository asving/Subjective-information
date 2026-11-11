# A race between circuits

*Which circuit does a network learn first, and when? Notes from a discussion between Asvin G. and Claude (Fable 5.2), November 2026. Every number below is printed by a script in `race/`.*

## The puzzle

Train a small transformer on a simple sequence task and its loss curve is often a staircase. Nothing happens for a long time. Then a circuit appears and the loss drops, and after another wait a second circuit refines the first.

Here is the case this note follows. The data are twelve tokens R R X R R X …, where R is a fair coin and X is the XOR of the two tokens before it. A one-layer transformer, trained by gradient descent from small weights, does three things that ask for an explanation.

- **It waits, and the wait varies.** The loss stays at 1 bit per token, the loss of guessing, until a time between 244 and 1,408 depending on the random seed.
- **It learns a rule that is half wrong.** The first circuit predicts the XOR of the last two tokens at every position, including the positions where the next token is a coin flip.
- **It corrects the rule on a different clock.** Restricting the prediction to XOR positions follows a few hundred time units later, however long the first wait was.

This note gives a model that accounts for all three with nothing fitted. It then asks how far the model carries: to the optimiser people actually use, and to a setting from the literature.

**Where this sits.** Alternating Gradient Flows (Kunin, Marchetti et al., 2025) describes training as dormant neurons accumulating evidence until one activates. We take that alternation and extend it from single neurons to circuits whose parts depend on each other. We then test the result on the leap staircase of Abbe, Boix-Adserà and Misiakiewicz (2023).

**What to watch.** One number runs through everything: k, the number of weights that must grow together before a circuit does anything. Under plain gradient descent the wait multiplies with each one. Under Adam with clean gradients it stops depending on k. With noisy gradients the dependence returns.

## The mechanism in miniature

The simplest object with a plateau is a product of weights. Fit the product of k weights to the constant 1, starting each weight at x₀ = 0.1. The gradient on each weight is the error times the product of the others, so a weight can grow only as fast as the others have already grown.

For k = 2 this is exponential growth, and the wait is log(1/x₀). For k ≥ 3 the growth is slower at first, and the wait is about x₀^(2−k). In general, if g is the size of the error signal, the product reaches full size after

    h_k(x₀) / g       h_1 = 1 − x₀,    h_2 = log(1/x₀),    h_k = (x₀^(2−k) − 1) / (k − 2)  for k ≥ 3

Adam behaves differently. It divides each gradient by its recent size, so every weight moves about one learning rate per step however small its gradient is. The wait is then a distance to travel.

The table gives the time until the product passes half its target (`unit_price.py chain`).

| k | h_k(0.1) | Plain gradient descent | Adam |
|---|---|---|---|
| 2 | 2.3 | 2.3 | 0.54 |
| 3 | 9.0 | 9.2 | 0.56 |
| 4 | 49.5 | 49.8 | 0.58 |
| 6 | 2,499.7 | 2,500.1 | 0.61 |

Each extra weight costs plain gradient descent a factor of about 1/x₀. It costs Adam almost nothing.

## From a chain to a circuit

A circuit in a network is such a chain wherever its output is a product of weights along a path. To use this we need names for the pieces.

**Units.** The residual stream has one column per feature and one row per position. A *unit* reads some columns and writes one. A head is a unit, a neuron is a unit, and so is a hand-written piece of circuit such as "multiply the current token by the previous one". A unit that reads another unit's column depends on it.

**Factors.** Each unit contributes a read and a write, and along a path from the input to the prediction these multiply. The weights on the path that are still at their initial size x₀ are its *dormant factors*, and k is their number. A unit whose factors have all grown is *active*.

**Gain.** The *gain* g of a dormant unit is the correlation of its output, at unit size, with the current prediction error. It plays the part of the error signal in the chain.

The wait for a unit is then h_k(x₀)/g. The race is the alternation of Alternating Gradient Flows: dormant units accumulate gain, the first to finish becomes active, the active units refit, and the error they leave sets the gains for the next round.

For a single neuron this is that paper's theory. The extension matters in a case the theory cannot reach. There, a neuron whose output has no correlation with the error never activates. But a head that only copies a token is such a unit, and networks learn them, because a later unit uses the copy. Seen as a path, the difficulty goes away: the head's weights are extra factors of the unit that reads it. Three consequences follow.

- **Bundles.** A unit that is useless alone is paid for inside the bundle that uses it.
- **Depth multiplies.** Under plain gradient descent each dormant factor beyond the second multiplies the wait by about 1/x₀.
- **Reuse is cheap.** A unit that reads active units pays only for its own factors. A new read on an active neuron has no wait.

## The puzzle, answered

Return to the transformer of the opening (`gd_rrxor.py`: one head, six ReLU neurons, every weight a scalar, all starting at scale ε).

**Which circuit comes first.** No single earlier token says anything about the next one, because any two tokens of a block are independent. The first useful circuit must combine two tokens. The shortest path that does so is a bundle: the head copies earlier tokens into a column, and a neuron reads that column together with the current token. It has three dormant factors: the head's write, the neuron's read of the copy, and the neuron's write. The neuron's read of the current token does not count, because a ReLU needs only its sign.

**How long it waits, and why seeds differ.** Three factors give a wait of about 1/(g·ε). To do better than "about", we integrate the three-factor equations for all six neurons from the network's own initial weights. The gains come from the data and from the inputs on which each neuron starts out active; for seed 0 they are 0, 0.006, 0.023, 0, −0.023 and 0.011. Nothing is fitted (`unit_price.py plateau`).

| Initial scale ε | Actual end of plateau, 7 seeds | Predicted / actual |
|---|---|---|
| 0.01 | 2,159 to 13,144 | 0.97 to 1.02 |
| 0.1 | 244 to 1,408 | 0.74 to 1.33 |

The spread across seeds is the spread in initial weights and gains. The equations keep only the leading order in ε, and the fit is looser at 0.1.

**Why the rule is half wrong.** Nothing in the bundle reads position, so its output is the same function of the last two tokens everywhere. The network predicts the XOR with probability about 0.72 at XOR positions and 0.68 elsewhere. A rule of this kind still pays: at its best it brings the loss from 1.000 to 0.892, against an optimum of 0.636.

**Why the correction runs on a different clock.** The correction is a read of position on a neuron that is already active. That is one new factor on an active unit, so it has no plateau. It follows 220 to 366 time units after the first drop for ε from 0.3 to 0.01, while the plateau itself grows from 138 to 3,249.

## The optimiser people use

A reader will object that nobody trains transformers with plain gradient descent. The chain already says what to expect from Adam: the wait should depend neither on k nor on ε.

It does not. Under Adam with learning rate 0.01 (`unit_price.py adam`), the same transformer leaves its plateau at step 54 to 56 for ε from 0.3 to 0.01, a range over which the wait under plain gradient descent grows more than twentyfold. Seven seeds give 46 to 68 steps, apart from one that takes over a thousand. The half-wrong stage is still there but shorter.

This changes what decides the order of circuits. If every unit travels at the same speed whatever its k, the unit that matters first is the one with the cleanest gradient: the largest gain relative to its spread from token to token. That ranking has no prices and no free parameter.

We tried it on a harder version of the task, RRXOR started at a random phase, where a predictor has to infer the phase from the history (`unit_price.py order`).

- **Stage 1.** The XOR head, current token times previous token, ranks first of 3,738 candidate units and linked pairs.
- **Stage 2.** A bundle ranks first of 17,415: a slot marking where the XOR rule was violated, plus the XOR multiplied by an OR of violations every third position back. This is the circuit reported for trained transformers (A.G.). A lookup on its slots scores 0.8245 bits per token against an optimum of 0.8234.

Plain gradient descent would order things differently. By a rough count the bundle has ten factors, and a cheap alternative, the token two back, has four. The bundle would come first only if weights started above 0.81 of their full size. So the order seen in trained transformers is itself evidence for the flat price.

**How we got here.** This ranking was not the starting point. We first ran the race with hand-set prices of about 3 bits per column a unit names, under two rules: a wait proportional to the price, and a wait exponential in it. On random-start RRXOR the first rule put the phase circuit after 465 tokens, right behind the XOR, and the second after 75,000,000. Only the first matches what networks do, and at the time we could not say why. The chain says why: the exponential rule is the multiplying law of plain gradient descent, and the proportional rule is far closer to Adam's flat price.

Run on five processes, the proportional rule builds the circuits reported for trained transformers.

| Process | Circuit the race builds | Reported for trained transformers (A.G.) | Script |
|---|---|---|---|
| RRXOR, aligned start | Soft XOR at every position (probability 0.70), then position gates | Soft XOR, then gating (above) | `rrxor_aligned.py` |
| RRXOR, random start | XOR, then an OR over the history of XOR violations | XOR early, then that phase circuit | `rrxor_random.py` |
| Orbit v, Uv, U²v, … of an orthogonal matrix | Previous-token head plus similarity head: loss 0.495, optimum 0.270 | Fit the current token on earlier ones and shift | `orbit.py` |
| Mess3 | Token tally with strengths 1.05, 0.54, 0.31 on the last three tokens; belief readable with R² 0.985 | Belief state linearly readable (Shai et al.) | `mess3.py` |
| Self-inoculation | Swap applied everywhere at half strength, then a switch on the situation belief (flip 0.94 in A, 0.06 in B), and the blind swap rolled back from 0.53 to 0.02 | Detour through a shared flip, then a switch | `self_inoculation.py` |

The half-wrong XOR is one instance of a pattern in this table: a cheap rule is applied everywhere at partial strength, a gate arrives, and the ungated strength returns to zero. Only the soft XOR and the orbit circuit were computed before the trained result was known.

## A test in someone else's setting

Everything so far is on tasks we chose. The leap staircase is an established result about what gradient descent learns and when, and it is stated in terms close to ours.

Abbe, Boix-Adserà and Misiakiewicz (2023) train a two-layer network on targets that are sums of monomials in a few coordinates of x ∈ {+1, −1}^d. A monomial that brings in L new coordinates is a *leap* of L. They find that a leap of L takes about d^(L−1) steps of online SGD, and that a staircase such as z₁ + z₁z₂ + z₁z₂z₃ is learnt fast because each term brings in one coordinate.

In our terms a unit is a pair (neuron, monomial). Its dormant factors are the neuron's output weight and its first-layer weights on the new coordinates, so a leap of L is L + 1 factors. Its gain is the monomial's coefficient times the mean L-th derivative of the activation. The staircase is reuse. `leap.py` tests this, first with exact gradients over all 2^d inputs and then in the paper's own regime of fresh samples.

**The plateau, and a correction.** The factor equations alone put the end of the plateau too early, at as little as 0.27 of the actual time for L = 4. What they miss is particular to sigmoid neurons. Such a neuron emits output before it has learnt anything, that output is not part of the target, and the loss pulls back the weights that produce it. Adding this pull-back, which has no free parameter, gives the last column. We found it after the failure; it was not predicted.

| Neurons, d | L | Actual end of plateau, 3 seeds | Factors alone / actual | With pull-back / actual |
|---|---|---|---|---|
| 16, 10 | 2 | 189 to 283 | 0.76 to 0.86 | 1.02 to 1.05 |
| 16, 10 | 3 | 560 to 1,979 | 0.59 to 0.79 | 0.96 to 1.00 |
| 16, 10 | 4 | 11,389 to 16,901 | 0.27 to 0.60 | 0.93 to 1.02 |
| 64, 8 | 2 | 89 to 154 | 0.86 to 0.91 | 1.00 to 1.02 |
| 64, 8 | 3 | 376 to 671 | 0.80 to 0.86 | 0.94 to 0.96 |
| 64, 8 | 4 | 2,957 to 5,286 | 0.64 to 0.69 | 0.86 to 0.91 |

**The staircase is reuse.** Under plain gradient descent the staircase reaches z₁z₂z₃ at time 56 to 66. The monomial z₁z₂z₃ alone takes 558 to 854. Each step of the staircase adds one factor to neurons that are already active.

**Adam with exact gradients is nearly flat.** The plateau lasts 29 to 82 steps for L = 2, 3 and 4 from initial scales 0.3 and 0.1. The staircase gives Adam no advantage: both targets take about 80 to 120 steps. There is one exception, at L = 4 from scales 0.03 and 0.01, where the plateau lasts 146 to 287 steps.

**Fresh samples bring the dependence back.** The paper's d^(L−1) is a statement about online SGD, where every step sees new data, so this is the comparison that matters. The table gives the median steps over three seeds until the test loss is 20% under that of predicting zero, with 64 neurons and batch 32 (`leap.py online`).

| Target | Optimiser | d = 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| z₁z₂ | SGD | 600 | 1,400 | 2,100 | 3,600 | 9,300 | |
| z₁z₂ | Adam | 400 | 500 | 600 | 900 | 1,500 | |
| z₁z₂z₃ | SGD | 1,400 | 3,000 | 6,700 | 178,700 | 1,023,900 | |
| z₁z₂z₃ | Adam | 500 | 700 | 1,300 | 3,300 | 120,700 | about 2,700,000 |

At leap 2, SGD's steps grow as d^1.07 from d = 64 to 256, which is the paper's d^(L−1), and Adam's as d^0.66. At leap 3 both rise far more steeply than d² at these sizes, SGD between d = 64 and 128 and Adam one doubling later. At d = 512 one Adam seed of three did not finish within 3,000,000 steps, and at d = 1,024 none did.

Larger batches shorten the wait for both. At d = 128 and leap 3, Adam takes 10,600, 3,300, 1,100 and 500 steps at batch 8, 32, 128 and 512, and SGD takes 276,500, 178,700, 22,200 and 26,500.

Our reading is that Adam can rescale a small gradient but cannot remove its noise. With fresh samples the gradient of a dormant unit is a small signal, a product of L small weights, under noise that does not shrink with it. We did not measure this directly.

So the three statements of the opening stand as follows. The multiplying law of plain gradient descent holds in both settings: it predicts the plateau to within 3% for the transformer at ε = 0.01, and to within 14% for the leap staircase once the pull-back is included. The flat price of Adam holds, approximately, with clean gradients. With noisy gradients in high dimension neither optimiser escapes the cost of a leap, and at leap 3 Adam is ahead by about one doubling of the dimension.

## What is not established

- **Units are written by hand.** The model says when a given unit appears. It does not find the units.
- **No real transformer has been compared.** The networks here are a one-layer toy with scalar weights and a two-layer network. Layer norm and weight decay are untested.
- **The noisy regime is measured, not modelled.** The sharp rise at leap 3 comes from three seeds and one batch size per dimension, and the model has no law for it.
- **Adam's flat price is approximate.** It rises at leap 4 from small weights, and one toy-transformer seed in seven was slow. The cause of the rise is not identified; it is not Adam's small-gradient floor.
- **The five hand-priced races have not been rerun with derived waits.** Factor counts for hand-written units are rough, since a wiring does not say how many weights implement it.

## Reproducing

Plain Python 3 unless noted. Each script prints its own setup and results.

    cd race
    python3 unit_price.py          # about 1 min   the chain, the plateau, Adam, the race with no prices
    python3 gd_rrxor.py            # about 20 s    the toy transformer under plain gradient descent
    python leap.py                 # about 1 min   the leap staircase with exact gradients (needs numpy)
    python leap.py online          # about 1.5 h   the leap staircase with fresh samples (needs numpy)
    python3 rrxor_aligned.py       # about 1 s     the five hand-priced races
    python3 rrxor_random.py        # about 35 s
    python3 orbit.py               # about 10 s
    python3 mess3.py               # about 15 s
    python3 self_inoculation.py    # about 15 s

`torch_read_write.py` is an earlier measurement that this note no longer uses.

## References

- E. Abbe, E. Boix-Adserà, T. Misiakiewicz, *SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics*, arXiv:2302.11055, 2023.
- D. Kunin, G. L. Marchetti, F. Chen, D. Karkada, J. B. Simon, M. R. DeWeese, S. Ganguli, N. Miolane, *Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks*, NeurIPS 2025.
- A. Shai, S. Marzen, L. Teixeira, A. Gietelink Oldenziel, P. Riechers, *Transformers represent belief state geometry in their residual stream*, arXiv:2405.15943.
- Asvin G., *Self-Inoculation*, asving.com, September 2026.
