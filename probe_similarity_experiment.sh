#!/bin/bash

# Per-layer experiments
uv run probe_similarity_experiment.py -m olmo_model -t standard control -pt lr -e per_layer_two_metrics -sh -sv
uv run probe_similarity_experiment.py -m tiny_aya_global -t standard control -pt lr -e per_layer_two_metrics -sh -sv

# Refitted probe compared with original probe experiments
# uv run probe_similarity_experiment.py -m olmo_model tiny_aya_global -t standard -pt lr -e per_extra_iter -nr 1 -ir 2 -sh -sv
# uv run probe_similarity_experiment.py -m olmo_model tiny_aya_global -t standard -pt lr -e per_extra_iter -nr 1 -ir 1000 -sh -sv
# exec $SHELL; # Turn this on to keep the terminal from automatically closing