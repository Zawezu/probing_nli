#!/bin/bash
# uv run experiment_1.py -pt mm 
# uv run experiment_2.py -pt mm -nr 1
# uv run experiment_1.py -pt lr 
uv run experiment_2.py -pt lr -ir 1
exec $SHELL; # Turn this on to keep the terminal from automatically closing