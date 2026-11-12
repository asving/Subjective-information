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

Two earlier results point the same way on the optimiser. Kunstner et al. (2023) report that full-batch Adam behaves like sign descent and that sign descent degrades at small batch sizes, and Kou et al. (2024) show that sign SGD with large batches learns sparse parities at the statistical-query limit. We take both from a literature search and have not worked through them. What this note adds is the count: which weights are still small, and what each one costs under each optimiser.

**What to watch.** One number runs through everything: k, the number of weights that must grow together before a circuit does anything. Under plain gradient descent the wait multiplies with each one. Under Adam with clean gradients it stops depending on k. With noisy gradients Adam's speed is set by the gradient's signal-to-noise ratio, and the dependence on k returns.

## The mechanism in miniature

The simplest object with a plateau is a chain of weights.

**The chain.** Let w_1, …, w_k be real numbers, the *weights*, and set

    output      f = w_1 · w_2 · … · w_k
    loss        ℓ = ½ (f − 1)²
    training    dw_i/dt = −∂ℓ/∂w_i = (1 − f) · ∏_{j ≠ i} w_j,        w_i(0) = x₀

This is a network stripped down to one path. The input is the constant 1, each of k layers multiplies it by one trainable number, and training tries to make the output 1. The last line says that a weight can grow only as fast as the others have already grown. While all of them are small, all of them grow slowly, and that is the plateau.

**The wait.** Start all weights equal, so that they stay equal to a common value x, and replace the error 1 − f by a constant g. Then

    dx/dt = g · x^(k−1),     and x reaches 1 at time    T = h_k(x₀) / g
    h_1 = 1 − x₀,    h_2 = log(1/x₀),    h_k = (x₀^(2−k) − 1) / (k − 2)   for k ≥ 3

In the chain the error is close to 1 until f stops being small, so T with g = 1 is the length of the plateau. For k = 2 the growth is exponential, and T is logarithmic in 1/x₀. For k ≥ 3 the growth starts slower, and T is a power of 1/x₀ that rises with k.

**Adam.** Write G for the gradient ∂ℓ/∂w of one weight. Adam keeps running averages of G and of G² and steps by their ratio:

    m ← 0.9 m + 0.1 G,     v ← 0.999 v + 0.001 G²,     w ← w − η · m̂ / (√v̂ + 10⁻⁸)

where m̂ and v̂ are m and v corrected for their start at zero. While G keeps its sign and changes slowly, m̂/√v̂ is close to ±1, so the weight moves about η per step however small G is. The wait is then the distance the weights must travel divided by η, and k drops out.

The table gives the time, in steps times learning rate, until f passes ½ (`unit_price.py chain`).

| k | h_k(0.1) | Plain gradient descent | Adam |
|---|---|---|---|
| 2 | 2.3 | 2.3 | 0.54 |
| 3 | 9.0 | 9.2 | 0.56 |
| 4 | 49.5 | 49.8 | 0.58 |
| 6 | 2,499.7 | 2,500.1 | 0.61 |

Each extra weight costs plain gradient descent a factor of about 1/x₀. It costs Adam almost nothing.

## From a chain to a circuit

A circuit in a network behaves like the chain wherever its output is a product of weights along a path. Here is the statement, followed by what its pieces mean in a network.

**The model.** A *unit* u consists of a finite set F(u) of weights, its *factors*, and a function φ_u of the input, its *output shape*. It adds to the network's prediction the amount

    ( ∏_{f ∈ F(u)} w_f ) · φ_u(input)

Let δ(input) be the derivative of the loss with respect to the prediction, at the current network. The *gain* of u is

    g_u = −E[ δ · φ_u ]

To first order in the amounts added by units that are still small, the loss is L₀ − Σ_u g_u ∏_{f ∈ F(u)} w_f, and gradient descent on it is

    dw_f/dt = Σ_{u : f ∈ F(u)} g_u · ∏_{f' ∈ F(u), f' ≠ f} w_f'

For a single unit this is the chain with g_u in place of the error. So a unit with k factors, all starting at size x₀ with signs that make g_u ∏ w_f positive, becomes large at time h_k(x₀)/|g_u|.

**Units in a network.** The residual stream has one column per feature and one row per position. A unit reads some columns and writes one. A head is a unit, a neuron is a unit, and so is a hand-written piece of circuit such as "multiply the current token by the previous one". Its output shape φ_u is what it would add to the prediction if all its factors were 1, with the units it reads held as they are.

**Factors in a network.** Each unit on a path from the input to the prediction contributes a read and a write, and these are the factors that multiply. In the networks tested here each factor is a single number. In a wider network a read or a write is a vector, and the factor is its size along the direction the unit uses; the end of the next section tests the count of factors in that case. Factors still at their initial size x₀ are *dormant*, and k from now on counts only those. A unit whose factors have all grown is *active*. Two units can share a factor, as when several neurons read one head; the sum over u in the last display is for that.

**The race.** Dormant units follow the last display. The first to become large is active. The active units then refit, which changes δ and with it every gain, and the next round begins. This is the alternation of Alternating Gradient Flows, and for a single neuron the model is that paper's theory.

The extension matters in a case that theory cannot reach. There, a neuron whose output has no correlation with the error has zero gain and never activates. But a head that only copies a token is such a unit, and networks learn them, because a later unit uses the copy. In the model the difficulty goes away: the head's weights are members of F(u) for the unit u that reads it. Three consequences follow.

- **Bundles.** A unit that is useless alone is paid for inside the bundle that uses it, as extra factors.
- **Depth multiplies.** Under plain gradient descent each dormant factor beyond the second multiplies the wait h_k(x₀)/g by about 1/x₀.
- **Reuse is cheap.** A unit that reads active units pays only for its own factors, since the active ones no longer count towards k. A new read on an active neuron is k = 1 and has no plateau.

## The puzzle, answered

Return to the transformer of the opening. Tokens are x_t = ±1 at positions t = 1, …, 12, and the network is (`gd_rrxor.py`)

    attention    α(t, ·) = softmax over t' ∈ {t, …, t−3} of   b[t − t'] + β · x_t · x_t'
    head         A_t = ω · Σ_t' α(t, t') · x_t'
    neuron j     n_j = ReLU( u_j · x_t + v_j · A_t + q_j[t] + c_j )            j = 1, …, 6
    logit        z_t = Σ_j r_j · n_j + ρ_x · x_t + ρ_A · A_t + d[t] + bias

Every symbol other than x and t is one scalar weight, drawn from a Gaussian of scale ε. The loss is the log loss of predicting the next token from z_t, averaged exactly over all sequences and positions, and training is full-batch gradient descent.

**Which circuit comes first.** No single earlier token says anything about the next one, because any two tokens of a block are independent. The first useful circuit must combine two tokens. The shortest path that does so runs through the head and one neuron. For neuron j it is the unit

    factors         F = { ω, v_j, r_j }
    output shape    φ_j = [ u_j · x_t + q_j[t] + c_j > 0 ] · ā_t,       ā_t = mean of those of x_t, …, x_(t−3) that exist
    gain            g_j = E[ (y − ½) · φ_j ],                          y = 1 if the next token is +1, else 0

and the general equations become

    dr_j/dt = g_j · v_j · ω,      dv_j/dt = g_j · r_j · ω,      dω/dt = Σ_j g_j · r_j · v_j

The head copies earlier tokens into the column A, which is useless alone, and the neuron reads the copy: a bundle with three factors. The neuron's read u_j of the current token is not a factor. It enters φ_j only through the bracket, which says on which inputs the neuron starts out active, and the bracket's overlap with ā_t is what makes φ_j carry the product of the last two tokens.

**How long it waits, and why seeds differ.** With three factors the wait is about h_3(ε)/g, or 1/(g·ε). To do better than "about", we integrate the three displayed equations for all six neurons from the network's own initial weights. The gains are computed from the data and the initial brackets; for seed 0 they are 0, 0.006, 0.023, 0, −0.023 and 0.011. Nothing is fitted (`unit_price.py plateau`).

| Initial scale ε | Actual end of plateau, 7 seeds | Predicted / actual |
|---|---|---|
| 0.01 | 2,159 to 13,144 | 0.97 to 1.02 |
| 0.1 | 244 to 1,408 | 0.74 to 1.33 |

The spread across seeds is the spread in initial weights and gains. The equations keep only the leading order in ε, and the fit is looser at 0.1.

**Why the rule is half wrong.** Nothing in φ_j reads position, so the bundle computes the same function of the last two tokens everywhere. The network predicts the XOR with probability about 0.72 at XOR positions and 0.68 elsewhere. A rule of this kind still pays: at its best it brings the loss from 1.000 to 0.892, against an optimum of 0.636.

**Why the correction runs on a different clock.** The correction is growth in q_j[t], the read of position on a neuron that is already active. That is one new factor on an active unit, so k = 1 and there is no plateau. It follows 220 to 366 time units after the first drop for ε from 0.3 to 0.01, while the plateau itself grows from 138 to 3,249.

**The same law with vector weights.** In the network above every factor is one number. To test the count of factors where reads and writes are vectors, take an ordinary one-layer transformer on the same data (`real_transformer.py scale`): width 32, two heads, 64 ReLU neurons, learnt position embeddings, no layer norm, exact gradients, and every weight matrix drawn as a Gaussian of standard deviation 0.1·s. The path of the first circuit crosses six matrices that start small:

    token embedding → V → O → MLP in → MLP out → unembedding

The attention pattern starts uniform, so the query and key matrices are not on the path, and the MLP's direct read of the current token again enters only through a sign. Scaling every matrix by s scales each of the six factors by s. The model therefore predicts a wait going as h_6, which for small s is s^(−4). We wrote this down before the run.

| s | Step of the first drop under plain gradient descent, 3 seeds | Local exponent, seed 0 | Under Adam |
|---|---|---|---|
| 1.4 | 51, 51, 67 | | 36 to 38 |
| 1.0 | 138, 140, 186 | 2.96 | 44 to 46 |
| 0.7 | 413, 452, 645 | 3.07 | 52 to 53 |
| 0.5 | 1,453, 1,620, 2,447 | 3.74 | 57 to 59 |
| 0.35 | 5,865, 6,588, 10,545 | 3.91 | 61 to 64 |
| 0.25 | 22,289, 25,382, 41,049 | 3.97 | 65 to 67 |
| 0.18 | 82,864 (one seed) | 4.00 | 72 to 74 |

The exponent climbs to 4 as s shrinks. The last column belongs to the next section.

## The optimiser people use

A reader will object that nobody trains transformers with plain gradient descent. The chain already says what to expect from Adam: the wait should depend neither on k nor on ε.

It does not. Under Adam with η = 0.01 (`unit_price.py adam`), the same transformer leaves its plateau at step 54 to 56 for ε from 0.3 to 0.01, a range over which the wait under plain gradient descent grows more than twentyfold. Seven seeds give 46 to 68 steps, apart from one that takes over a thousand. The half-wrong stage is still there but shorter. With vector weights the same holds: over the range of s in which the wait under plain gradient descent grows 1,600-fold, the wait under Adam doubles (last column of the table above).

This changes what decides the order of circuits. If every unit travels at the same speed whatever its k, the unit that matters first is the one whose gradient is cleanest. Precisely, for a candidate unit with output shape φ, with p the current predicted probability and y the outcome, define

    score(φ) = ( E[ (y − p) · φ ] )² / E[ p (1 − p) · φ² ]

The numerator is the squared gain. The denominator is the token-to-token variance of the quantity (y − p)·φ whose mean is the gain. The score is therefore a squared signal-to-noise ratio per token, and it has no prices and no free parameter.

We tried this ranking on a harder version of the task, RRXOR started at a random phase, where a predictor has to infer the phase from the history (`unit_price.py order`).

- **Stage 1.** The XOR head, current token times previous token, ranks first of 3,738 candidate units and linked pairs.
- **Stage 2.** A bundle ranks first of 17,415: a slot marking where the XOR rule was violated, plus the XOR multiplied by an OR of violations every third position back. This is the circuit reported for trained transformers (A.G.). A lookup on its slots scores 0.8245 bits per token against an optimum of 0.8234.

Plain gradient descent would order things differently. By a rough count the bundle has ten factors, and a cheap alternative, the token two back, has four. Comparing h_10(x₀)/g with h_4(x₀)/g for the two gains, the bundle comes first only if x₀ is above 0.81. So the order seen in trained transformers is itself evidence for the flat price.

**A real transformer agrees.** `real_transformer.py phase` trains a four-layer transformer (width 64, four heads, layer norm, initial standard deviation 0.02) on the same random-phase data with exact gradients. Under Adam with η = 0.001, in three seeds the loss passes 0.97 at steps 37, 58 and 90, goes under the XOR-only level of 0.926 by steps 90, 160 and 226, and is within 0.02 of the optimum by steps 261, 428 and 466. The second stage begins about two and a half times as far in as the first. We read only the loss levels; we did not extract the circuit.

Plain SGD from the same initial weights learns nothing in 6,000 steps at learning rate 0.3, and diverges at 1.0. From weights five and ten times larger, where the factors are no longer small, SGD learns both stages in the same order, with the later ones relatively slower than under Adam.

| Initial std | Optimiser | Loss under 0.97 | Under 0.91 | Under 0.84 |
|---|---|---|---|---|
| 0.1 | Adam, η = 0.001 | 96 | 195 | 420 |
| 0.1 | SGD, η = 0.1 | 199 | 618 | 2,398 |
| 0.2 | Adam, η = 0.001 | 70 | 203 | 368 |
| 0.2 | SGD, η = 0.1 | 131 | 710 | 2,461 |

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

**The setting** (Abbe, Boix-Adserà and Misiakiewicz, 2023). The input x is uniform on {+1, −1}^d. The target is y = Σ_S c_S ∏_{i ∈ S} x_i, a sum over a few sets S of coordinates. The learner is

    network     f(x) = Σ_j a_j · σ( ⟨w_j, x⟩ + b_j ),        σ(z) = 1 / (1 + e^(0.5 − z))
    loss        ½ E ( f(x) − y )²

A term whose set S brings in L coordinates not used by earlier terms is a *leap* of L. The paper finds that a leap of L takes about d^(L−1) steps of online SGD, and that a staircase such as x_1 + x_1 x_2 + x_1 x_2 x_3 is learnt fast because each term brings in one coordinate.

**The unit.** For one term with set S of size L and coefficient c, neuron j gives the unit

    factors         F = { a_j } ∪ { w_ji : i ∈ S }                              L + 1 factors
    output shape    φ_j = σ^(L)( Σ_{i ∉ S} w_ji x_i + b_j ) · ∏_{i ∈ S} x_i     σ^(L) the L-th derivative
    gain            g_j = c · E[ σ^(L)( Σ_{i ∉ S} w_ji x_i + b_j ) ]

and the general equations become

    da_j/dt = g_j · ∏_{i ∈ S} w_ji,        dw_ji/dt = a_j · g_j · ∏_{l ∈ S, l ≠ i} w_jl

So a leap of L is L + 1 dormant factors, and the staircase is reuse: after one term is learnt, its coordinates are active and the next term has one dormant factor per neuron. `leap.py` tests this, first with exact gradients over all 2^d inputs and then in the paper's own regime of fresh samples.

**The plateau, and a correction.** These equations alone put the end of the plateau too early, at as little as 0.27 of the actual time for L = 4. To see what they miss, expand the loss:

    ½ E (f − y)² = ½ E y² − E[ y f ] + ½ E f²

The unit equations are gradient descent on the middle term, to leading order in the w_ji. The last term is the network's own output. Write V for the matrix of the w_ji with i ∈ S, and σ_j for σ(Σ_{i ∉ S} w_ji x_i + b_j). To second order in V,

    ½ E f² = ½ aᵀ K_V a,        K_V = K + Q ∘ V Vᵀ + ½ (N R + Rᵀ N)
    K_jl = E[ σ_j σ_l ],    Q_jl = E[ σ'_j σ'_l ],    R_jl = E[ σ''_j σ_l ],    N = diag( |V_1|², …, |V_M|² )

with ∘ the entrywise product and V_j the j-th row of V. The *pull-back* is gradient descent on this term too. In words: a sigmoid neuron emits output before it has learnt anything, that output is not part of the target, and the loss pulls back the weights that produce it. Nothing in K_V is fitted. We found the pull-back after the failure; it was not predicted.

| Neurons, d | L | Actual end of plateau, 3 seeds | Unit equations / actual | With pull-back / actual |
|---|---|---|---|---|
| 16, 10 | 2 | 189 to 283 | 0.76 to 0.86 | 1.02 to 1.05 |
| 16, 10 | 3 | 560 to 1,979 | 0.59 to 0.79 | 0.96 to 1.00 |
| 16, 10 | 4 | 11,389 to 16,901 | 0.27 to 0.60 | 0.93 to 1.02 |
| 64, 8 | 2 | 89 to 154 | 0.86 to 0.91 | 1.00 to 1.02 |
| 64, 8 | 3 | 376 to 671 | 0.80 to 0.86 | 0.94 to 0.96 |
| 64, 8 | 4 | 2,957 to 5,286 | 0.64 to 0.69 | 0.86 to 0.91 |

Here the target is the single term x_1 ⋯ x_L, the w_ji start as Gaussians of scale 0.1, the a_j of scale 0.01 and the b_j at zero, the learning rate is 0.2, and the plateau ends when the loss is 5% under ½ E y².

**The staircase is reuse.** Under plain gradient descent the staircase x_1 + x_1 x_2 + x_1 x_2 x_3 reaches its last term at time 56 to 66. The term x_1 x_2 x_3 alone takes 558 to 854.

**Adam with exact gradients is nearly flat.** The plateau lasts 29 to 82 steps for L = 2, 3 and 4 from initial scales 0.3 and 0.1. The staircase gives Adam no advantage: both targets take about 80 to 120 steps. There is one exception, at L = 4 from scales 0.03 and 0.01, where the plateau lasts 146 to 287 steps.

**Fresh samples, both layers trained.** The paper's d^(L−1) is a statement about online SGD, where every step sees new data, so this is the comparison that matters. Each step draws a fresh batch of 32 inputs. There are 64 neurons, the w_ji start at scale 1/√d and the a_j at 0.1, and the learning rate is 0.1 for SGD and 0.003 for Adam. The table gives the median over three seeds of the number of steps until the loss on 4,096 held-out inputs is 20% under ½ E y² (`leap.py online`).

| Target | Optimiser | d = 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|
| x_1 x_2 | SGD | 600 | 1,400 | 2,100 | 3,600 | 9,300 | |
| x_1 x_2 | Adam | 400 | 500 | 600 | 900 | 1,500 | |
| x_1 x_2 x_3 | SGD | 1,400 | 3,000 | 6,700 | 178,700 | 1,023,900 | |
| x_1 x_2 x_3 | Adam | 500 | 700 | 1,300 | 3,300 | 120,700 | about 2,700,000 |

At leap 2, SGD's steps grow as d^1.07 from d = 64 to 256, which is the paper's d^(L−1), and Adam's as d^0.66. At leap 3 both rise far more steeply than d² at these sizes, SGD between d = 64 and 128 and Adam one doubling later. At d = 512 one Adam seed of three did not finish within 3,000,000 steps, and at d = 1,024 none did.

**The steep rise is the pull-back.** We first read that rise as the cost of noise. Three observations say otherwise (`leap.py pullback`; leap 3, batch 32).

- **The weights shrink during the plateau.** At d = 128, in an SGD run that escapes at step 197,500, the root-mean-square output weight falls from 0.103 to 0.045 within 1,000 steps and to 0.030 by step 150,000. The first-layer weights on the monomial's coordinates fall to half their initial size.
- **A lower learning rate on the output weights removes most of the wait.** At d = 128, cutting that rate from 0.1 to 0.025 brings SGD's median from 178,700 steps to 22,800.
- **A wider network removes more.** At d = 128 and the lower rate, 256 neurons take 8,600 steps. At d = 256 Adam's median falls from 120,700 steps with 64 neurons to 39,900 with 256 and 24,800 with 1,024.

Our reading goes back to K_V. Its leading part has rank about d + 1, a constant mode and one mode per input coordinate, so M neurons leave about M − d − 1 directions of the output weights that the pull-back does not touch. With 64 neurons these run out as d passes 64, which is where SGD's steep rise began.

**With the output weights frozen, two clean laws.** Holding the a_j at their initial values removes the pull-back on them. It is also the first phase of the algorithm the paper analyses. The tables give median steps over three seeds, with 64 neurons (`leap.py frozen`).

| Batch 32 | d = 32 | 64 | 128 | 256 | 512 | Exponent in d |
|---|---|---|---|---|---|---|
| Leap 3, SGD | 7,700 | 17,600 | 18,000 | 19,200 | 31,300 | 0.42 |
| Leap 3, Adam | 1,300 | 2,100 | 2,900 | 4,200 | 12,300 | 0.75 |
| Leap 2, SGD | 3,600 | 3,300 | 4,000 | 4,900 | 5,400 | 0.17 |
| Leap 2, Adam | 700 | 700 | 900 | 1,100 | 1,400 | 0.27 |

| Leap 3, d = 256 | Batch 8 | 32 | 128 | 512 | 2,048 | Exponent in batch |
|---|---|---|---|---|---|---|
| SGD | 26,400 | 19,200 | 19,500 | 20,000 | 19,800 | −0.04 |
| Adam | 10,000 | 4,200 | 2,000 | 1,000 | 600 | −0.51 |

SGD follows the multiplying law and ignores the noise. With the a_j fixed at an appreciable size, the dormant factors are the L first-layer weights, of size x₀ = 1/√d. The law gives a wait going as h_3(x₀), that is d^0.5, at leap 3, and as h_2(x₀), that is log d, at leap 2. The measured exponents are 0.42 and 0.17, and the batch size makes no difference.

Adam's speed is the signal-to-noise ratio of the gradient. Let the gradient of a dormant factor, over the draw of a batch of size B, have mean μ and standard deviation ρ/√B. When μ√B/ρ is below 1, the ratio m̂/√v̂ in Adam's update averages about μ√B/ρ, so the factor moves η·μ√B/ρ per step where clean gradients gave η. The wait should then go as 1/√B, and the measured exponent is −0.51. Because μ is g times the product of the other dormant factors, the dependence on k and x₀ comes back with the noise. We have not derived Adam's exponent in d, which is higher than SGD's at both leaps.

That difference in exponents predicts that plain SGD overtakes Adam when d is large and the batch small. It does (`leap.py crossover`). At batch 8 and d = 256, Adam takes 10,000 steps and SGD 26,400. At d = 1,024, SGD takes 269,800, and none of three Adam seeds finishes within 2,000,000. We predicted the reversal before running it, from the fitted exponents. It came out far stronger than they implied, so Adam's wait is not a power law in d here.

None of these runs shows d^(L−1), which at leap 3 would be d². Two differences from the paper may account for that. Its theorem lets the learning rate shrink as a power of d, and ours is constant. And its bound is a statistical one, which at these sizes sits below the waits we measure: at d = 512 SGD's 31,300 steps of batch 32 are about a million samples, against d² of about 260,000.

So the three statements of the opening stand as follows. The multiplying law of plain gradient descent holds in every setting tried: it predicts the plateau to within 3% for the toy transformer at ε = 0.01, gives the exponent 4 for six matrices in a transformer with vector weights, and predicts the leap plateau to within 14% once the pull-back is included. The flat price of Adam holds, approximately, with clean gradients. With noisy gradients Adam's speed is set by the signal-to-noise ratio, so its wait goes as 1/√B here, while SGD's does not depend on the batch. The largest effect in the noisy runs was neither the optimiser nor the noise. It was the pull-back on the output weights.

## What is not established

- **Units are written by hand.** The model says when a given unit appears. It does not find the units.
- **Real transformers are compared only from outside.** We checked a scaling exponent in one and the loss levels another passes through. No circuit was read out of their weights. Weight decay is untested, and layer norm appears only in the four-layer run.
- **The noisy laws are partial.** Adam's dependence on batch size is measured and explained. Its exponent in d under noise is not derived. The account of the steep rise by the pull-back rests on three observations at one leap and two dimensions.
- **Adam's flat price is approximate.** It rises at leap 4 from small weights, and one toy-transformer seed in seven was slow. The cause of the rise is not identified; it is not the 10⁻⁸ in Adam's denominator.
- **The five hand-priced races have not been rerun with derived waits.** Factor counts for hand-written units are rough, since a wiring does not say how many weights implement it.

## Reproducing

Plain Python 3 unless noted. Each script prints its own setup and results.

    cd race
    python3 unit_price.py          # about 1 min   the chain, the plateau, Adam, the race with no prices
    python3 gd_rrxor.py            # about 20 s    the toy transformer under plain gradient descent
    python leap.py                 # about 1 min   the leap staircase with exact gradients (needs numpy)
    python leap.py online          # about 1.5 h   the leap staircase with fresh samples (needs numpy)
    python leap.py pullback        # about 4 min   why the wait rises steeply
    python leap.py frozen          # about 5 min   output weights held fixed
    python leap.py crossover       # about 30 min  SGD overtakes Adam at large d and small batch
    python real_transformer.py scale small   # about 30 min   vector weights, initial scale (needs PyTorch)
    python real_transformer.py phase sgd     # about 1 h      four layers, random-start RRXOR (needs PyTorch)
    python3 rrxor_aligned.py       # about 1 s     the five hand-priced races
    python3 rrxor_random.py        # about 35 s
    python3 orbit.py               # about 10 s
    python3 mess3.py               # about 15 s
    python3 self_inoculation.py    # about 15 s

`torch_read_write.py` is an earlier measurement that this note no longer uses.

## References

- E. Abbe, E. Boix-Adserà, T. Misiakiewicz, *SGD learning on neural networks: leap complexity and saddle-to-saddle dynamics*, arXiv:2302.11055, 2023.
- Y. Kou, Z. Chen, Q. Gu, S. Kakade, *Matching the Statistical Query Lower Bound for k-Sparse Parity Problems with Sign Stochastic Gradient Descent*, NeurIPS 2024, arXiv:2404.12376.
- F. Kunstner, J. Chen, J. W. Lavington, M. Schmidt, *Noise Is Not the Main Factor Behind the Gap Between SGD and Adam on Transformers, but Sign Descent Might Be*, ICLR 2023, arXiv:2304.13960.
- D. Kunin, G. L. Marchetti, F. Chen, D. Karkada, J. B. Simon, M. R. DeWeese, S. Ganguli, N. Miolane, *Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks*, NeurIPS 2025.
- A. Shai, S. Marzen, L. Teixeira, A. Gietelink Oldenziel, P. Riechers, *Transformers represent belief state geometry in their residual stream*, arXiv:2405.15943.
- Asvin G., *Self-Inoculation*, asving.com, September 2026.
