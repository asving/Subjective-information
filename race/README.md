# race/

The scripts behind `../circuit-race.md`. Every number in the note is printed by one of them, and the printed outputs are in `outputs/`, one file per command.

## Starting cold

1. Read `../circuit-race.md`. Its opening has the picture in brief, and its sections "What is not established" and "Where next" say what is open.
2. To check a number, search `outputs/` for it. The outputs print numbers without thousands separators and with a plain minus sign (13144 and -0.29, where the note has 13,144 and −0.29), and the note often rounds or gives the range of a printed column. To regenerate everything, run `PY=/path/to/python bash run_all.sh`, with `PY` a Python that has numpy and PyTorch (about two hours on ten cores; `bash run_all.sh quick` does the plain-Python jobs in a few minutes).
3. To race a new menu of units, use `Race` in `unit_race.py`. A unit is a tuple `(name, factors, shape)`: `factors` is a tuple of weight names, and `shape` is a list of `(cell index, value)`. `phase_menu()` is the shortest worked example. `readout.py` shows how to fit such shapes to a transformer's logit during training.

## Interpreters

- Plain Python 3, no packages: `gd_rrxor.py`, `unit_price.py`, `unit_race.py` and the five order-only races.
- numpy: `leap.py`.
- PyTorch and numpy: `real_transformer.py`, `readout.py`.

## What each script backs

| Script and part | Section of the note | Output file | Time |
|---|---|---|---|
| `unit_price.py chain` | The model, step 2; The optimiser: Adam with exact gradients | `unit_price.txt` | seconds |
| `unit_price.py plateau` | One circuit: how long it waits | `unit_price.txt` | 1 min |
| `unit_price.py adam` | The optimiser: Adam with exact gradients | `unit_price.txt` | seconds |
| `unit_price.py order` | The optimiser: which unit is worth most | `unit_price.txt` | 35 s |
| `unit_price.py shapes` | One circuit: a unit has two shapes | `unit_price.txt` | seconds |
| `unit_price.py pool` | Several circuits: the candidates these units come from | `unit_price.txt` | 35 s |
| `gd_rrxor.py` | The puzzle; One circuit (the toy transformer itself) | `gd_rrxor.txt` | 20 s |
| `real_transformer.py scale small` | One circuit: the same law with vector weights | `real_transformer_scale_small.txt` | 20 min |
| `leap.py` | One circuit: a plateau from the literature; The optimiser (the leap setting with exact gradients) | `leap.txt` | seconds |
| `unit_race.py` | Several circuits: two circuits that share a weight | `unit_race.txt` | 1.5 min |
| `unit_race.py trace` | The same, one run in detail (not quoted in the note) | `unit_race_trace.txt` | seconds |
| `unit_race.py phase` | Several circuits: a circuit written from primitives; The optimiser: the race under Adam | `unit_race_phase.txt` | seconds |
| `unit_race.py count` | The optimiser: does the count of weights matter | `unit_race_count.txt` | seconds |
| `rrxor_aligned.py`, `rrxor_random.py`, `orbit.py`, `mess3.py`, `self_inoculation.py` | Appendix B (order-only races with hand-set prices); `rrxor_aligned.py` also gives the losses 0.892 and 0.636 quoted in One circuit | one file each | 1 to 45 s |
| `leap.py online`, `pullback`, `frozen`, `crossover` | Appendix A; the batch-size table is also in The optimiser: Adam with noisy gradients | `leap_online.txt` and so on | 6 min to 2 h |
| `real_transformer.py phase sgd` | Against a real transformer: the network | `real_transformer_phase_sgd.txt` | 1.5 h |
| `readout.py`, `readout.py sgd`, `readout.py extra` | Against a real transformer: the readout | `readout.txt`, `readout_sgd.txt`, `readout_extra.txt` | 7 to 11 min each |
| `readout.py sweep` | Against a real transformer: the slow later stages | `readout_sweep.txt` | 1 h |
| `real_transformer.py rate` | The same: the learning rate does not set the pace | `real_transformer_rate.txt` | under 1 min |

## How the scripts depend on each other

- `unit_price.py` and `unit_race.py` import `gd_rrxor.py` (the toy transformer and its data) and `rrxor_random.py` (the data and unit language of random-start RRXOR).
- `readout.py` imports `real_transformer.py` (the four-layer transformer) and `rrxor_random.py` (the shapes).
- `rrxor_random.py` defines the candidate units used by `unit_price.py order` and `pool`: a head `(pattern, query column, pooled column, pool)` and a gate. Its hand-set prices are used only by its own race.

The times are those of the saved outputs, produced with five jobs running at once.

## Conventions

- Names differ in a few places between the note and the code. The note's λ (how sticky the coin is) is `s` in `unit_race.py`. The neuron's read of position, q_j[t], is `gate`. In the phase circuit the note's ρ_e, ρ_τ and τ are the factors `r_e`, `r_s` and `s`.
- `unit_price.py order` prints an older, rougher count of weights ("about 10 factors"); the race in `unit_race.py phase` uses the leaner count described in the note.
- The predictions that the note says were written before a run were recorded in the authors' working log at the time. That log is not in this repository.
- `real_transformer.Transformer(norm=False)` removes every layer norm. Nothing in the note uses it yet; it is there for the next experiment on Adam's clock.

- Tokens are ±1 inside the race scripts and 0/1 inside `real_transformer.py`. The XOR of two ±1 tokens is minus their product.
- Losses are in bits per token. "Time" is steps times learning rate under plain gradient descent, and a step count under Adam.
- Runs are deterministic given the seed. `readout.py` uses four CPU threads and `real_transformer.py` one or two; the step counts in the note were produced with those settings.

`archive/torch_read_write.py` is an earlier measurement that the note no longer uses.
