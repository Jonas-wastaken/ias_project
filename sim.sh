#!/bin/bash

# Define the path to the Python script
PYTHON_SCRIPT="./src/sim.py"
NUM_INTERSECTIONS=$((RANDOM % 101 + 200)) # -i
NUM_BORDERS=$((NUM_INTERSECTIONS * (RANDOM % 3 + 2))) # -b
NUM_CARS=$((NUM_INTERSECTIONS * (RANDOM % 6 + 5))) # -c
MIN_DISTANCE=$((RANDOM % 16 + 5)) # -min
MAX_DISTANCE=$((MIN_DISTANCE * (RANDOM % 3 + 2))) # -max
STEPS=10000 # -s

# Print the generated values (optional, for debugging or verification)
echo "NUM_INTERSECTIONS: $NUM_INTERSECTIONS"
echo "NUM_BORDERS: $NUM_BORDERS"
echo "NUM_CARS: $NUM_CARS"
echo "MIN_DISTANCE: $MIN_DISTANCE"
echo "MAX_DISTANCE: $MAX_DISTANCE"
echo "STEPS: $STEPS"

# Check if the Python script exists
if [ -f "$PYTHON_SCRIPT" ]; then
    # Execute the Python script with the arguments
    echo "Executing $PYTHON_SCRIPT..."
    python3 "$PYTHON_SCRIPT" -i "$NUM_INTERSECTIONS" -b "$NUM_BORDERS" -c "$NUM_CARS" -min "$MIN_DISTANCE" -max "$MAX_DISTANCE" -s "$STEPS"
else
    echo "Error: $PYTHON_SCRIPT does not exist."
fi
