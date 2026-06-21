#!/bin/bash
# uv run experiment_1.py -pt mm 
# uv run experiment_2.py -pt mm -nr 1
# uv run experiment_1.py -pt lr 
# uv run experiment_2.py -pt lr -nr 2 -ir 2
# uv run experiment_2.py -pt lr -nr 2 -ir 1000

# Japanese original label ablation
# uv run experiment_1.py -pt mm -fol -l jp
# uv run experiment_1.py -pt lr -fol -l jp
uv run experiment_2.py -pt mm -fol -nr 1
uv run experiment_2.py -pt lr -fol -nr 2 -ir 2
uv run experiment_2.py -pt lr -fol -nr 2 -ir 1000
exec $SHELL; # Turn this on to keep the terminal from automatically closing