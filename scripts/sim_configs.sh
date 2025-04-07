#!/bin/bash
# execute from working directory

NUM_INTERSECTIONS=("25" "50" "100")
NUM_BORDERS=("25" "50" "100")
NUM_CARS=("25" "50" "100")
STEPS="1000"
OPTIMIZATION_TYPE="advanced_ml"

for intersections in "${NUM_INTERSECTIONS[@]}"; do
    for borders in "${NUM_BORDERS[@]}"; do
        python scripts/sim.py --num_intersections "$intersections" --num_borders "$borders" --num_cars "50" --min_distance "9" --max_distance "15" --steps "$STEPS" --optimization_type "$OPTIMIZATION_TYPE"
    done
done

for cars in "${NUM_CARS[@]}"; do
    python scripts/sim.py --num_intersections "25" --num_borders "50" --num_cars "$cars" --min_distance "2" --max_distance "8" --steps "$STEPS" --optimization_type "$OPTIMIZATION_TYPE"
done

for cars in "${NUM_CARS[@]}"; do
    python scripts/sim.py --num_intersections "25" --num_borders "50" --num_cars "$cars" --min_distance "9" --max_distance "15" --steps "$STEPS" --optimization_type "$OPTIMIZATION_TYPE"
done

for cars in "${NUM_CARS[@]}"; do
    python scripts/sim.py --num_intersections "25" --num_borders "50" --num_cars "$cars" --min_distance "16" --max_distance "22" --steps "$STEPS" --optimization_type "$OPTIMIZATION_TYPE"
done

