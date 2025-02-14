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
        self.agent = self.model.agents[0]
        self.assertIsInstance(self.agent, CarAgent)
        logging.info(f"Agent {self.agent.unique_id} initialized")
        logging.info(f"Agent start: {self.agent.start}")
        logging.info(f"Agent position: {self.agent.position}")
        logging.info(f"Agent goal: {self.agent.goal}")
        logging.info(f"Agent path: {self.agent.path}")
        logging.info("Setup complete: CarAgent initialized")

    def test_compute_goal(self):
        logging.info("Test compute_goal")
        try:
            self.assertTrue(self.agent.goal.startswith("border"))
            self.assertNotEqual(self.agent.goal, self.agent.start)
            logging.info("Passed test_compute_goal")
        except AssertionError as e:
            logging.error(f"Failed test_compute_goal: {e}")
            raise

    def test_compute_path(self):
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
