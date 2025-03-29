import os
import unittest
import sys
from log_config import setup_logging
import re
import random
from math import sqrt

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from graph import Graph

logger = setup_logging("test_light")


class TestLightAgent(unittest.TestCase):
    # TODO: Setup the test environment
    # TODO: testen, ob jede Ampel genau eine open_lane hat und ob diese in den neighbor_lights enthalten ist
    # TODO: testen, ob cooldown beim ändern von open_lane beachtet wird

    # TODO später: update_waiting_cars testen, ob waiting_cars richtig aktualisiert wird

    pass
