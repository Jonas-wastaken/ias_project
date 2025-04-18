import os
import unittest
import sys
from log_config import setup_logging
import random

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from model import TrafficModel
from car import CarAgent
from graph import Graph
from light import LightAgent

logger = setup_logging("test_model")


class TestTrafficModel(unittest.TestCase):
    def setUp(self):
        self.num_cars = 5
        self.num_intersections = 10
        self.num_borders = 3
        self.min_distance = 2
        self.max_distance = 10
        self.model = TrafficModel(
            num_cars=self.num_cars,
            num_intersections=self.num_intersections,
            num_borders=self.num_borders,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
        )
        logger.info("Setup complete: TrafficModel initialized")

    def test_initial_cars(self):
        """Test initial CarAgents.

        - This test checks if the cars are initialized correctly.
            - Check if the cars are instances of the CarAgent class.
            - Check if the cars paths are initialized as dictionaries
            - Assert that the number of cars matches the expected number.

        Raises:
            AssertionError: If the cars are not instances of the CarAgent class, the cars paths are not initialized as dictionaries, or the number of cars does not match the expected number.
        """
        logger.info("Test initial cars")
        logger.info(f"Initialized {self.num_cars} CarAgents!")
        try:
            for car in self.model.get_agents_by_type("CarAgent"):
                logger.info(f"Agent {car.unique_id}")
                logger.info(f"{self.model.car_paths[car.unique_id]}")
                self.assertIsInstance(car, CarAgent)
                self.assertIsInstance(self.model.car_paths[car.unique_id], dict)
            self.assertEqual(
                len(self.model.get_agents_by_type("CarAgent")), self.num_cars
            )
            self.assertEqual(len(self.model.car_paths), self.num_cars)
            logger.info("Passed test_initial_cars")
        except AssertionError as e:
            logger.error(f"Failed test_initial_cars: {e}")
            raise

    def test_initial_lights(self):
        """Test initial LightAgents.

        - This test checks if the lights are initialized correctly.
            - Check if the lights are instances of the LightAgent class.
            - Assert that the number of lights matches the expected number (= the number of intersections).
            - Check if each intersection has a LightAgent and each LightAgent is placed on an intersection.

        Raises:
            AssertionError: If the lights are not instances of the LightAgent class, or the number of lights does not match the expected number, or if each intersection does not have a LightAgent or if each LightAgent is not placed on an intersection.
        """
        logger.info("Test initial lights")
        logger.info(f"Initialized {self.num_intersections} LightAgents!")
        try:
            for light in self.model.get_agents_by_type("LightAgent"):
                logger.info(f"Light {light.unique_id}")
                self.assertIsInstance(light, LightAgent)
            self.assertEqual(
                len(self.model.get_agents_by_type("LightAgent")), self.num_intersections
            )
            lights_positions = [
                light.position for light in self.model.get_agents_by_type("LightAgent")
            ]
            self.assertEqual(
                lights_positions, self.model.grid.get_nodes("intersection")
            )
            logger.info("Passed test_initial_lights")
        except AssertionError as e:
            logger.error(f"Failed test_initial_lights: {e}")
            raise

    def test_graph_initialization(self):
        """Test graph initialization.

        - This test checks if the graph is initialized correctly.
            - Check if the grid is an instance of the Graph class.
            - Assert that the number of nodes in the graph matches the expected number of intersections and borders.

        Raises:
            AssertionError: If the grid is not an instance of the Graph class or the number of nodes in the graph does not match the expected number of intersections and borders.
        """
        logger.info("Test graph initialization")
        logger.info(f"Graph nodes: {self.model.grid.nodes}")
        try:
            self.assertIsInstance(self.model.grid, Graph)
            self.assertEqual(
                len(self.model.grid.nodes),
                self.num_intersections + self.num_borders,
            )
            logger.info("Passed test_graph_initialization")
        except AssertionError as e:
            logger.error(f"Failed test_graph_initialization: {e}")
            raise

    def test_car_step(self):
        """Test step.

        - This test checks if the cars move to their next positions after a step.
            - Save the initial positions of the cars.
            - Call the step method of the model.
            - Assert that the initial positions are not the same as the current positions.

        Raises:
            AssertionError: If the initial positions are the same as the current positions.
        """
        logger.info("Test step")
        initial_positions = self.model.car_paths.copy()
        logger.info(f"{[path for path in self.model.car_paths.values()]}")
        try:
            self.model.step()
            self.assertIsNot(initial_positions, self.model.car_paths)
            logger.info("Passed test_step")
        except AssertionError as e:
            logger.error(f"Failed test_step: {e}")
            raise

    def test_create_cars(self):
        """Test adding cars.

        Raises:
            AssertionError: If the number of cars is not increased by the expected number
        """
        logger.info("Testing adding car")
        initial_num_cars = len(self.model.get_agents_by_type("CarAgent"))
        additional_num_cars = random.randint(3, 100)
        try:
            self.model.create_cars(additional_num_cars)
            self.assertEqual(
                (initial_num_cars + additional_num_cars),
                len(self.model.get_agents_by_type("CarAgent")),
            )
            logger.info("Passed test_create_cars")
        except AssertionError as e:
            logger.error(f"Failed to add cars: {e}")
            raise


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestTrafficModel("test_initial_cars"))
    suite.addTest(TestTrafficModel("test_initial_lights"))
    suite.addTest(TestTrafficModel("test_graph_initialization"))
    suite.addTest(TestTrafficModel("test_car_step"))
    suite.addTest(TestTrafficModel("test_create_cars"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
