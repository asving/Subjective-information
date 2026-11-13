#!/bin/bash
# Regenerate every file in race/outputs/, one per command quoted in circuit-race.md.
# About an hour and a half on ten cores; the longest single job is "leap.py online".
#
#   PY=/path/to/python bash run_all.sh        PY needs numpy and PyTorch (default: python)
#   bash run_all.sh quick                     only the jobs that need plain Python 3 (a few minutes)
#
# The jobs run in five lanes at once and use about ten cores between them.
cd "$(dirname "$0")" || exit 1
PY="${PY:-python} -W ignore"
mkdir -p outputs

run() {
    local out="outputs/$1.txt"
    shift
    "$@" > "$out" 2>&1
    echo "exit $? after ${SECONDS}s: $out"
}

(
    run gd_rrxor python3 gd_rrxor.py
    run unit_price python3 unit_price.py
    run unit_race python3 unit_race.py
    run unit_race_trace python3 unit_race.py trace
    run unit_race_phase python3 unit_race.py phase
    run unit_race_count python3 unit_race.py count
    run rrxor_aligned python3 rrxor_aligned.py
    run rrxor_random python3 rrxor_random.py
    run orbit python3 orbit.py
    run mess3 python3 mess3.py
    run self_inoculation python3 self_inoculation.py
) &
if [ "$1" = quick ]; then wait; exit 0; fi
(
    run readout $PY readout.py
    run readout_sgd $PY readout.py sgd
    run readout_extra $PY readout.py extra
    run readout_sweep $PY readout.py sweep
) &
( run leap_online $PY leap.py online ) &
(
    run real_transformer_scale_small $PY real_transformer.py scale small
    run real_transformer_rate $PY real_transformer.py rate
    run leap_crossover $PY leap.py crossover
    run leap_pullback $PY leap.py pullback
    run leap_frozen $PY leap.py frozen
    run leap $PY leap.py
) &
( run real_transformer_phase_sgd $PY real_transformer.py phase sgd ) &
wait
