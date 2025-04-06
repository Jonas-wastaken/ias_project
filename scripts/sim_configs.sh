#!/bin/bash
# execute from working directory

NUM_INTERSECTIONS=("25" "50" "100")
NUM_BORDERS=("25" "50" "100")
NUM_CARS=("25" "50" "100")
MIN_DISTANCE="2"
MAX_DISTANCE="8"
STEPS="1000"
OPTIMIZATION_TYPE="advanced_ml"

for intersections in "${NUM_INTERSECTIONS[@]}"; do
    for borders in "${NUM_BORDERS[@]}"; do
        for cars in "${NUM_CARS[@]}"; do
            python scripts/sim.py --num_intersections "$intersections" --num_borders "$borders" --num_cars "$cars" --min_distance "$MIN_DISTANCE" --max_distance "$MAX_DISTANCE" --steps "$STEPS" --optimization_type "$OPTIMIZATION_TYPE"
        done
    done
done

MIN_DISTANCE="9"
MAX_DISTANCE="15"

for intersections in "${NUM_INTERSECTIONS[@]}"; do
    for borders in "${NUM_BORDERS[@]}"; do
        for cars in "${NUM_CARS[@]}"; do
            python scripts/sim.py --num_intersections "$intersections" --num_borders "$borders" --num_cars "$cars" --min_distance "$MIN_DISTANCE" --max_distance "$MAX_DISTANCE" --steps "$STEPS" --optimization_type "$OPTIMIZATION_TYPE"
        done
    done
done

MIN_DISTANCE="16"
MAX_DISTANCE="22"

for intersections in "${NUM_INTERSECTIONS[@]}"; do
    for borders in "${NUM_BORDERS[@]}"; do
        for cars in "${NUM_CARS[@]}"; do
            python scripts/sim.py --num_intersections "$intersections" --num_borders "$borders" --num_cars "$cars" --min_distance "$MIN_DISTANCE" --max_distance "$MAX_DISTANCE" --steps "$STEPS" --optimization_type "$OPTIMIZATION_TYPE"
        done
    done
done