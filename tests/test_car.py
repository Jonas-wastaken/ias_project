import os
import unittest
import sys
import logging

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from car import CarAgent, AgentArrived
from model import TrafficModel
import re


log_dir = os.path.join("tests", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(log_dir, "test_car.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class TestCarAgent(unittest.TestCase):
    """Unit tests for the CarAgent class.

    This class contains unit tests for the CarAgent class, which is part of the TrafficModel.
    It tests the initialization, goal computation, path computation, movement, and exception handling of the CarAgent.

    Attributes:
        num_agents (int): Number of agents in the TrafficModel.
        num_intersections (int): Number of intersections in the TrafficModel.
        num_borders (int): Number of borders in the TrafficModel.
        min_distance (int): Minimum distance between intersections.
        max_distance (int): Maximum distance between intersections.
        model (TrafficModel): Instance of the TrafficModel.
        agent (CarAgent): Instance of the CarAgent being tested.
    """   
    def setUp(self):
        self.num_agents = 1
        self.num_intersections = 10
        self.num_borders = 3
        self.min_distance = 1
        self.max_distance = 10
        self.model = TrafficModel(
            num_agents=self.num_agents,
            num_intersections=self.num_intersections,
            num_borders=self.num_borders,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
        )
        self.agent = self.model.get_agents_by_type("CarAgent")[0]
        self.assertIsInstance(self.agent, CarAgent)
        logging.info(f"Agent {self.agent.unique_id} initialized")
        logging.info(f"Agent start: {self.agent.start}")
        logging.info(f"Agent position: {self.agent.position}")
        logging.info(f"Agent goal: {self.agent.goal}")
        logging.info(f"Agent path: {self.agent.path}")
        logging.info("Setup complete: CarAgent initialized")

    def test_compute_goal(self):
        """Test the compute_goal method of the CarAgent class.

        - Test if the goal is a border node.
        - Test if the goal is not the same as the start node.

        Raises:
            AssertionError: If the goal is not a border node or if the goal is the same as the start node.
        """        
        logging.info("Test compute_goal")
        try:
            self.assertTrue(self.agent.goal.startswith("border"))
            self.assertNotEqual(self.agent.goal, self.agent.start)
            logging.info("Passed test_compute_goal")
        except AssertionError as e:
            logging.error(f"Failed test_compute_goal: {e}")
            raise

    def test_compute_path(self):
        """Test the compute_path method of the CarAgent class.

        - Test if the path is a dictionary.
        - Test if the start node is the first node in the path.
        - Test if the goal node is the last node in the path.
        - Test if other nodes in the path are intersection nodes.

        Raises:
            AssertionError: If the path is not a dictionary, if the start node is not the first node in the path, if the goal node is not the last node in the path, or if other nodes in the path are not intersection nodes.
        """
        logging.info("Test compute_path")
        try:
            self.assertIsInstance(self.agent.path, dict)
            self.assertEqual(self.agent.start, list(self.agent.path.keys())[0])
            self.assertEqual(self.agent.goal, list(self.agent.path.keys())[-1])
            pattern = re.compile(r"^intersection_\d+$")
            self.assertTrue(all(pattern.match(node) for node in list(self.agent.path.keys())[1:-1]))
            logging.info("Passed test_compute_path")
        except AssertionError as e:
            logging.error(f"Failed test_compute_path: {e}")
            raise

    def test_move(self):
        """Test the move method of the CarAgent class.

        - Test if the agent moves to the next node in the path.

        Raises:
            AssertionError: If the agent does not move to the next node in the path
        """
        logging.info("Test move")
        try:
            for _ in range(self.agent.path[self.agent.start] + 1):
                self.agent.move()
            self.assertNotEqual(self.agent.position, self.agent.start)
            logging.info("Passed test_move")
        except AgentArrived:
            logging.info("Agent arrived at goal")
        except AssertionError as e:
            logging.error(f"Failed test_move: {e}")
            raise

    def test_agent_arrived_exception(self):
        """Test the AgentArrived exception of the CarAgent class.

        - Test if the exception is raised when the agent reaches it's goal.

        Raises:
            AssertionError: If the exception is not raised when the agent reaches it's goal
        """
        logging.info("Test agent_arrived_exception")
        try:
            self.agent.position = self.agent.goal
            with self.assertRaises(AgentArrived):
                self.agent.move()
            logging.info("Passed test_agent_arrived_exception")
        except AssertionError as e:
            logging.error(f"Failed test_agent_arrived_exception: {e}")
            raise


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCarAgent("test_compute_goal"))
    suite.addTest(TestCarAgent("test_compute_path"))
    suite.addTest(TestCarAgent("test_move"))
    suite.addTest(TestCarAgent("test_agent_arrived_exception"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
