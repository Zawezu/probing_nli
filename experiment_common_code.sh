#!/bin/bash
uv run probe_similarity_experiment.py -m olmo_model -pt mm -sf cos_sim -zwd 100 -e per_layer -pc -sv
uv run probe_similarity_experiment.py -m olmo_model -pt mm -sf maha_cos_sim -zwd 100 -e per_layer -pc -sv
uv run probe_similarity_experiment.py -m tiny_aya_global -pt mm -sf cos_sim -zwd 100 -e per_layer -pc -sv
uv run probe_similarity_experiment.py -m tiny_aya_global -pt mm -sf maha_cos_sim -zwd 100 -e per_layer -pc -sv
# exec $SHELL; # Turn this on to keep the terminal from automatically closing