# A race between circuits

*A toy model of which circuit a network learns first. Notes from a discussion between Asvin G. and Claude (Fable 5.2), November 2026. Every number below is printed by a script in `race/`.*

## The idea

The draft's emergence threshold says a bounded observer acquires a piece of structure once it has paid for itself: after ΔK/ΔH samples. Here the observer is a learner, the structures are circuits, and the question is concrete: given a data stream and a population of candidate units, which circuit forms first and what follows it?

The answer is computed, not trained. We enumerate small circuits built from a few transformer-like primitives, compute each one's exact benefit on the stream, and let them race. On the toy processes below the race builds the kind of circuit that small transformers are reported to learn. Where an order of appearance is known, one of the two population regimes tried (full attention) gets it right. The limits are listed at the end.

## The model

A population of units reads from and writes to one shared stream. There are three clocks: the position t in a sequence, the depth of a unit within one forward pass, and the number n of training samples seen so far.

**The stream.** A grid of slots. Row t belongs to position t. Column 0 holds the token; column i belongs to unit i.

**The population.** A unit is a *wiring* and nothing else: a tuple (a, b, c, P, pool) naming three columns it reads, a position pattern and a pooling mode. It has no weights of its own. At every position of every input, unit i reads the stream and writes its own column:

    s[t, i] = pool over t' in P(t) of  <s[t, a], s[t', b]> · s[t', c]

P is "here", a fixed offset, every p-th position back, all, or recency-weighted. The pool is a mean or its sharp limit, an OR. Any of a, b, c can be the constant 1, so copying a slot from elsewhere, matching two slots, and gating one slot by another at the same position are all special cases. A column holds a scalar or a small vector, and the bracket is a product or an inner product. The *parents* of a unit are the units whose columns it reads. Reads must not form a cycle, and a unit cannot read its own column, so there is no recursion. (The examples also use an indicator gate and, in the last one, a fixed permutation of a vector slot.)

In transformer terms a column is one feature direction in the residual stream. A unit is one rank-one piece of an attention head, or, when the pattern is "here", one neuron that multiplies features at the same position.

**One forward pass.** The only order is the one imposed by reads. A unit can fill its column, at all positions at once, as soon as its parents' columns are filled. Units that do not read each other act in parallel, and the depth of a unit is the longest chain of reads beneath it. A final reader predicts the next token from row t as a linear combination of its columns. Every unit fires on every input; what differs between units and inputs is how much they write.

**What a unit carries.** A wiring, and a number y in bits: its evidence. The wiring has a price κ in bits, a made-up description length of about 3 bits per column named plus a few for the pattern. Equivalently, the wiring has prior weight 2^(−κ) in the population. Evidence starts at −κ.

**Learning.** Each sample adds to a unit's evidence the bits it saved: the drop in loss from having that unit on, given its parents. A unit is *on* when its own evidence, plus the surplus of the units that read it, is positive and its parents are on. The reader's strengths adjust quickly, so they sit at their best values for whichever units are on; a strength can therefore fall back to zero when a later unit makes an earlier one redundant.

Globally this is a single statement. After n samples, the units that are on are the parent-closed set S that maximises

    n · ΔH(S) − κ(S)

where ΔH(S) is the loss that S removes (bits per token, with the reader refitted) and κ(S) is its total price. This is the most probable set under the prior 2^(−κ(S)) and the evidence of n samples; the local rule above is its exact form when benefits add and each unit has one parent. As n grows the maximiser gains units in bundles, and a bundle G enters when n·ΔH(G) = κ(G). Proposition 3 of the draft is the single-unit case. Weighting each set by 2^(−κ(S) − n·L(S)), with L its loss, gives the mixture over time used in example 5.

**Population size.** The statement above assumes a unit for every wiring, so that every wiring gathers evidence on every sample. Call this *full attention*. With a small population only the wirings currently held gather evidence, and a new concept needs an old unit rewired. If units are rewired toward wirings in proportion to current belief in them, the expected wait for a bundle is about (κ + (2^κ − 1)/ln 2)/ΔH, exponential in the price. Call this *search-limited*. It appears here only as that formula.

## Running a race

The scripts follow the maximiser greedily, which is the race.

1. Write down the process and compute the exact distribution over contexts (or sample it).
2. Fix the primitives and prices.
3. At each stage, score every new unit, and every pair in which the second unit reads the first, by refitting the reader exactly. Adopt the bundle with the smallest κ/ΔH and repeat. The search-limited wait is printed alongside.

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

## A transformer-faithful version, trained by gradient descent

`gd_rrxor.py` drops the race and trains a one-layer transformer on aligned RRXOR. Every unit is one scalar weight of an attention head, or one ReLU neuron.

    score         s(t, t') = b[t − t'] + β · x_t · x_t'                 t' = t, …, t−3
    attention     α(t, ·) = softmax of the scores                        the only pool is this weighted mean
    head output   A_t = ω · Σ_t' α(t, t') · x_t'
    neuron j      n_j = ReLU(u_j · x_t + v_j · A_t + g_j[t] + c_j)
    logit         z_t = Σ_j r_j · n_j + ρ_x · x_t + ρ_A · A_t + d[t]

Here b is a relative-position bias (one weight per offset), β a content weight, ω the value/output weight, g_j[t] the neuron's read of position t, and r, ρ, d the unembedding. All weights start as Gaussians of scale ε and follow full-batch gradient descent on the exact loss.

**The gradients.** Write δ_t for the prediction error at position t, and G_t = δ_t · (ρ_A + Σ_j r_j · v_j · [n_j > 0]) for the error signal arriving at the head's output. Then, averaging over data and positions:

    unembedding   dr_j/dn = −E[ δ_t · n_j ]
    neuron input  du_j/dn = −E[ δ_t · r_j · [n_j > 0] · x_t ]            (same for v_j with A_t, and for g_j[t])
    value/output  dω/dn   = −E[ G_t · ā_t ]                              ā_t = Σ_t' α(t, t') · x_t'
    position      db_δ/dn = −E[ G_t · ω · α(t, t−δ) · (x_(t−δ) − ā_t) ]
    content       dβ/dn   = −E[ G_t · ω · x_t · Var_α(x_t') ]

Every rate is a correlation with the current error, multiplied by the other weights on the same path: downstream weights scale the error signal, upstream weights scale the feature. Three things follow.

- **Plateaus and two growth laws.** If k weights must grow together from scale ε, each grows at a rate proportional to the product of the others. For k = 2 the growth is exponential and the wait is about log(1/ε)/c. For k = 3 it is hyperbolic and the wait is about 1/(c·ε). With price κ = log₂(1/ε), these are the full-attention and the search-limited rule: "tested in proportion to belief" is exactly a rate proportional to the weight's own size.
- **Reuse.** A circuit that shares grown weights with an earlier one needs fewer new weights, so it follows a faster law.
- **Rollback.** The correlation is with the current error. Once another path explains it, the sign flips and the weight shrinks.

**The run.** The network sits on a plateau, then passes through the phase-blind soft XOR, then gates it.

| Initial scale ε | Plateau ends at | ε × time | Soft stage: P(XOR) at XOR positions / elsewhere | Gating follows after |
|---|---|---|---|---|
| 0.3 | 138 | 41.5 | 0.724 / 0.672 | 222 |
| 0.1 | 314 | 31.4 | 0.718 / 0.676 | 220 |
| 0.03 | 1,060 | 31.8 | 0.719 / 0.680 | 287 |
| 0.01 | 3,249 | 32.5 | 0.716 / 0.681 | 366 |

The plateau scales as 1/ε, the three-weight law: r_j, v_j and ω grow together, and the inverse of their product's cube root falls roughly linearly in time. Gating arrives a nearly fixed time later. Over seven seeds at ε = 0.1, the time to the soft XOR has standard deviation over mean 0.73, and the further time to gating 0.23. The race model predicted both the soft stage (0.70 everywhere) and this contrast in spread.

Columns here are given feature directions, there is one head, and there is no layer norm.

**Reads and writes.** A neuron reads the stream through one number and writes one direction. Forward, its write cannot depend on any distinction its read ignores. Backward, the dual holds exactly: the gradient on the write is the error signal averaged over the inputs where the neuron is active, and does not involve the write itself; the gradient on the read sees the error only through its component along the write.

`torch_read_write.py` measures the resulting order on a real one-layer transformer (width 32, 64 neurons, Adam, small initial weights, five seeds). The leading neurons' writes come within cosine 0.9 of their final directions at steps 75, 150, 98, 309 and 68; their reads at steps 185, 228, 100, 315 and 301. Writes are never later and are clearly earlier in three seeds. The final writes all lie along the logit direction (cosine 0.89 to 0.99), while the reads spread over several directions. Neither group carries the gain alone: putting all reads, or all writes, back to their initial values returns the loss to 1.000 at every stage.

## Closest prior work: Alternating Gradient Flows

Kunin, Marchetti and coauthors (NeurIPS 2025) approximate gradient flow in two-layer networks with small initial weights by an alternation. Every neuron starts *dormant*. A dormant neuron climbs its *utility*, the correlation of its output with the current residual, and accumulates it over time. The first neuron whose accumulated utility passes a threshold set by its initial norm becomes *active*. The active neurons then minimise the loss together, which fixes a new residual, and the cycle repeats.

The correspondence with this note is close.

| Alternating Gradient Flows | This note |
|---|---|
| Dormant and active neurons | Units that are off and on |
| Utility: correlation of a neuron's output with the residual | Benefit: the loss a unit removes once the reader is refitted |
| Accumulated utility against a threshold −log(initial norm) | Evidence against a price |
| Order κ of the utility (2 for a linear or ReLU neuron, 3 for a quadratic neuron or a rank-one attention head) | Number of weights that must grow together, giving the same two waiting laws |
| Cost minimisation over the active neurons | Refitting the reader |

The differences run both ways. Their neurons find their own direction by gradient ascent on the utility, so there is no menu of wirings and no made-up price, and they prove convergence to gradient flow for diagonal linear networks. They treat two-layer networks only, where neurons interact only through the residual; a unit with no utility of its own can never activate, so parents and bundles lie outside the framework. Twice the neuron is not their unit either: for fully connected linear networks they switch to singular modes, and for modular addition they activate a group of neurons per frequency.

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
- **The unit simplifies attention in three ways.** In a transformer the position pattern and the content match are two terms of one learned score, the score passes through a softmax, and the pool is always a weighted mean. Here the pattern is chosen from a menu, the match multiplies the output directly, and the OR is a separate pool, standing for the sharp limit of the softmax.
- **Missing primitives.** There is no exact solve or normalisation, no recursion, and no model of an MLP as a lookup table.

## Reproducing

Plain Python 3, no dependencies. Each script prints its own setup, candidates and results.

    cd race
    python3 rrxor_aligned.py       # about 1 s   (--no-match: about 10 s)
    python3 rrxor_random.py        # about 35 s
    python3 orbit.py               # about 10 s
    python3 mess3.py               # about 15 s
    python3 self_inoculation.py    # about 15 s
    python3 gd_rrxor.py            # about 20 s   (gradient descent, not a race)
    python torch_read_write.py     # about 30 s   (needs PyTorch)

## References

- D. Kunin, G. L. Marchetti, F. Chen, D. Karkada, J. B. Simon, M. R. DeWeese, S. Ganguli, N. Miolane, *Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks*, NeurIPS 2025.
- G. Weiss, Y. Goldberg, E. Yahav, *Thinking Like Transformers* (RASP), ICML 2021.
- N. Elhage et al., *A Mathematical Framework for Transformer Circuits*, 2021.
- H. Zhou et al., *What Algorithms can Transformers Learn? A Study in Length Generalization* (the RASP-L conjecture), 2023.
- A. Shai, S. Marzen, L. Teixeira, A. Gietelink Oldenziel, P. Riechers, *Transformers represent belief state geometry in their residual stream*, arXiv:2405.15943.
- Asvin G., *Self-Inoculation*, asving.com, September 2026.
