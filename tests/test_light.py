import os
import unittest
import sys
import logging
import re
import random
from math import sqrt

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from graph import Graph

log_dir = os.path.join("tests", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_dir, "test_graph.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# TODO: schreib Tests ()