# A race between circuits

*A toy model of which circuit a network learns first. Notes from a discussion between Asvin G. and Claude (Fable 5.2), November 2026. Every number below is printed by a script in `race/`.*

## The idea

The draft's emergence threshold says a bounded observer acquires a piece of structure once it has paid for itself: after ΔK/ΔH samples. Here the observer is a learner, the structures are circuits, and the question is concrete: given a data stream and a pool of candidate units, which circuit forms first and what follows it?

The answer is computed, not trained. We enumerate small circuits built from a few transformer-like primitives, compute each one's exact benefit on the stream, and let them race. On the toy processes below the race builds the kind of circuit that small transformers are reported to learn. Where an order of appearance is known, one of the two race rules tried gets it right. The limits are listed at the end.

## The model

**Stream.** Every position holds a set of slots: the current token, and whatever units have written there.

**Units.** A unit reads slots and writes one slot. One form has covered almost every unit that mattered:

    out(t) = pool over s in P(t) of  <a(t), b(s)> · c(s)

Here a, b, c are any slots (raw, written by another unit, or the constant 1), P is a position pattern (here, a fixed offset, every p-th position back, all, or recency-weighted), and the pool is a mean or its sharp limit (an OR). Copying, matching two tokens, and pooling evidence over the history are all special cases. The examples also use a pointwise gate (one slot times an indicator of another) and, in the last one, a fixed permutation of a vector slot.

**Price.** Each unit has a price κ in bits: a made-up description length, about 3 bits per slot named plus a few for the pattern.

**Circuit and readout.** A circuit is a set of units closed under "reads from". The prediction is a fitted linear combination of the slots. Strengths are free and are refitted whenever the circuit changes.

**Benefit.** The benefit ΔH of adding a bundle of units to a circuit is the drop in loss, in bits per token, after refitting.

**The race.** Two rules were tried.

- *Full attention.* Every candidate is tested on every sample, so evidence for a bundle accumulates at ΔH bits per token against a prior of 2^(−κ). The bundle enters after κ/ΔH tokens, smallest first. Proposition 3 of the draft is the single-unit case.
- *Search-limited.* A candidate is tested in proportion to current belief in it. The expected wait is about (κ + (2^κ − 1)/ln 2)/ΔH, exponential in the price.

**Over time.** For a nested chain of circuits, circuit k carries weight proportional to 2^(−κ_k − n·L_k) after n tokens, where L_k is its loss. This turns a chain into a training trajectory.

## Running a race

1. Write down the process and compute the exact distribution over contexts (or sample it).
2. Fix the primitives and prices.
3. At each stage, score every new unit, and every pair in which the second unit reads the first, by refitting the readout exactly. Adopt the best bundle and repeat.

## Examples

| Process | What the race builds | What trained transformers do (reported by A.G.) | Script |
|---|---|---|---|
| RRXOR, aligned start | Soft XOR at every position, then position gates | n/a | `rrxor_aligned.py` |
| RRXOR, random start | XOR, then an OR over the history of XOR violations | XOR early, then that phase circuit; optimal | `rrxor_random.py` |
| Orbit v, Uv, U²v, … | Previous-token head plus similarity head, then stacked copies | Fit the current token on earlier ones, predict by shifting | `orbit.py` |
| Mess3 | A recency-weighted token tally carrying the belief | Belief state linearly readable (Shai et al.) | `mess3.py` |
| Self-inoculation | Blind half-swap, then a switch on the situation belief, then rollback | Detour through a shared flip, then a switch | `self_inoculation.py` |

### 1. RRXOR, aligned start

Twelve tokens: R R X R R X …, with R a fair coin and X the XOR of the two tokens before it. Baseline 1.000 bits per token, optimum 0.636.

| Step | Unit | Price | Benefit | Loss |
|---|---|---|---|---|
| 1 | Match head: current token times previous token | 9.58 | 0.108 | 0.892 |
| 2 | Gate that slot at two XOR positions | 11.92 | 0.107 | 0.785 |
| 3 | Extend the gate to the other two | 6.92 | 0.149 | 0.636 |

Step 1 is a soft XOR: it predicts the XOR of the last two tokens with probability 0.70 at every position. No single earlier token helps, because any two tokens of a block are independent, so the first circuit has to be a two-token interaction. Once the gate exists the ungated strength refits to zero. With attention that can only copy (`--no-match`), the first circuit is still a soft XOR, built from a copy head and one neuron.

### 2. RRXOR, random start

The same process started at a random phase. Baseline 1.000, optimum 0.8234.

- **Stage 1.** The XOR head wins out of 42 units and 3,696 pairs: price 9.58, benefit 0.074, loss 0.926.
- **Stage 2.** Out of 129 units and 17,286 pairs, the best bundle under full attention is a *violation slot* (current token times the XOR written one back) plus the XOR times an OR of violations every third position back, starting two back. Price 21.17, benefit 0.0455, loss 0.880. The violation slot alone has zero benefit.
- **The two rules disagree.** Full attention: XOR after 129 tokens, the phase bundle after 465, a cheap trick (the token two back) after 707. Search-limited: 15,000 for XOR, 15,400 for the trick, 75,000,000 for the bundle. Only the first order matches what networks do.
- **The circuit is essentially optimal.** A lookup on the XOR and the three OR flags scores 0.8245; adding position mod 3 gives 0.8234 exactly, and the phase belief is then an exact function of those slots.

### 3. Orbit of an orthogonal matrix

x₀ = v, x₁ = Uv, x₂ = U²v, …, with a fresh random unit vector v and orthogonal U per sequence (dimension 4, ten tokens, squared error, 300 sampled sequences). Predicting zero scores 1; the optimum is 0.270.

Out of 11 units and 517 pairs, the best bundle is a previous-token head plus a similarity head: compare the current token with every stored previous token and average the tokens at those positions. It takes the loss to 0.495. The race then stacks three more heads of the same form, each reading the last one's output: 0.381, 0.357, 0.353.

The optimum is exactly the reported circuit: fit x_t on the earlier tokens and apply the coefficients to the shifted tokens. The race finds the shift and the first gradient step of the fit (raw similarities as coefficients). It has no primitive for an exact solve, so it stalls above the optimum.

### 4. Mess3

Mess3 with x = 0.05, α = 0.85; all 3,279 prefixes up to length 7. Uniform guess 1.585, optimum 1.163.

The race adopts the mean of all tokens, the current token, the token one back, the token two back, and then a gate for "the last two tokens agree". Loss falls 1.228, 1.187, 1.173, 1.171, 1.166. The fitted strengths on the current token and the two before it are 1.05, 0.54, 0.31: a leaky tally. The exact belief is linearly readable from the slots with R² 0.985. A single recency-weighted tally with decay 0.5 reaches loss 1.177 and R² 0.964 on its own.

The belief is tracked in parallel, by re-weighting past tokens. The language has no recursion: a unit cannot read its own output at an earlier position.

### 5. Self-inoculation

The setting of [the post](https://asving.com/blog/self-inoculation/): x from one Mess3, y from one of two situation processes, and during finetuning the x target is swapped in situation A only. The learner starts with the exact x tracker and the exact situation belief as given slots; pretraining is not simulated. Candidates are permutations of a vector slot and gates by a scalar slot.

| Finetuning on A and B | Flip in A | Flip in B | Loss |
|---|---|---|---|
| Pretrained | 0.00 | 0.00 | 1.582 |
| Tracker strength turned down (free) | 0.25 | 0.25 | 1.533 |
| Swap of the tracker output, everywhere, half strength | 0.50 | 0.50 | 1.480 |
| Gate: unswapped tracker off when the belief says A | 0.74 | 0.25 | 1.444 |
| Gate: swapped tracker on when the belief says A | 0.94 | 0.06 | 1.412 |

The blind swap's strength goes 0.53, 0.55, 0.02: it is rolled back once the switch exists. The final loss matches the exact ideal (1.412), and the flip tracks the situation belief (0.04, 0.23, 0.50, 0.76, 0.96 at beliefs near 0.05, 0.25, 0.5, 0.75, 0.95). Finetuning on A alone ends after one step: the swap at full strength, flip 1.00 in both situations, and no gate with any benefit. Over time, the mixed run's flip in B rises to a peak of 0.39 and returns to 0.06.

## What the examples suggest

- **One unit type goes a long way.** Match a slot here against a slot there, and use the match to weight a third slot. This is close to RASP's select and aggregate, and to query, key and value composition in the transformer-circuits framework.
- **Soft rule, then gate, then rollback.** A cheap rule is applied everywhere at partial strength, a gate arrives, and the ungated strength returns to zero. The phase-blind XOR and the detour through a shared flip are the same event.
- **Predict, compare, pool, gate.** The learned circuits apply a rule to the past, compare it with what happened, pool the comparison over the history, and use the result to weight the rule now. That is the observer model running inside the forward pass.
- **Full attention fits better.** Every unit in a network is active on every input. The rule that assumes this is the one that places the RRXOR phase circuit correctly.

## What is not established

- **Prices and primitives are chosen by hand.** Orderings that hinge on them are not predictions. Examples: whether the gated RRXOR circuit enters in one step (it does if gating costs under 22.7 bits; here 18.8), and whether Mess3 starts with a plain mean or a recency-weighted one.
- **Only some results were blind.** The soft XOR and the orbit circuit were computed before the trained result was known. The RRXOR phase circuit was found after the language was widened to let heads read written slots, and self-inoculation was run with the post in hand.
- **No timings have been checked.** The race gives an order and relative waits. Nothing here has been compared with a training curve.
- **The search is shallow.** Pairs are searched only at the first one or two stages. The sizes are small (12 tokens, dimension 4, 7-token contexts).
- **Missing primitives.** There is no exact solve or normalisation, no recursion, and no model of an MLP as a lookup table.

## Reproducing

Plain Python 3, no dependencies. Each script prints its own setup, candidates and results.

    cd race
    python3 rrxor_aligned.py       # about 1 s   (--no-match: about 10 s)
    python3 rrxor_random.py        # about 35 s
    python3 orbit.py               # about 10 s
    python3 mess3.py               # about 15 s
    python3 self_inoculation.py    # about 15 s

## References

- G. Weiss, Y. Goldberg, E. Yahav, *Thinking Like Transformers* (RASP), ICML 2021.
- N. Elhage et al., *A Mathematical Framework for Transformer Circuits*, 2021.
- H. Zhou et al., *What Algorithms can Transformers Learn? A Study in Length Generalization* (the RASP-L conjecture), 2023.
- A. Shai, S. Marzen, L. Teixeira, A. Gietelink Oldenziel, P. Riechers, *Transformers represent belief state geometry in their residual stream*, arXiv:2405.15943.
- Asvin G., *Self-Inoculation*, asving.com, September 2026.
