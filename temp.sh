#!/bin/bash

NUM_INTERSECTIONS="50"
NUM_BORDERS="150"
NUM_CARS="250"
MIN_DISTANCE="5"
MAX_DISTANCE="15"
STEPS="1000"
OPTIMIZATION_TYPES=("none" "simple" "advanced" "advanced_ml")

for opt_type in "${OPTIMIZATION_TYPES[@]}"; do
    python src/sim.py \
        -i "$NUM_INTERSECTIONS" \
        -b "$NUM_BORDERS" \
        -c "$NUM_CARS" \
        -m "$MIN_DISTANCE" \
        -x "$MAX_DISTANCE" \
        -s "$STEPS" \
        -o "$opt_type"
done
