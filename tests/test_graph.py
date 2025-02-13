import os
import unittest
import sys
import logging

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


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.num_intersections = 10
        self.num_borders = 3
        self.min_distance = 1
        self.max_distance = 10
        self.graph = Graph(
            self.num_intersections,
            self.num_borders,
            self.min_distance,
            self.max_distance,
        )
        logging.info("Setup complete: Graph initialized")

    def test_add_intersections(self):
        intersections = [
            node for node in self.graph.nodes if node.startswith("intersection")
        ]
        logging.info(f"Intersections added: {intersections}")
        try:
            self.assertEqual(len(intersections), self.num_intersections)
            logging.info("Passed test_add_intersections")
        except AssertionError:
            logging.error(
                f"Failed test_add_intersections: Expected {self.num_intersections}, got {len(intersections)}"
            )
            raise

    def test_add_borders(self):
        borders = [node for node in self.graph.nodes if node.startswith("border")]
        logging.info(f"Borders added: {borders}")
        try:
            self.assertEqual(len(borders), self.num_borders)
            logging.info("Passed test_add_borders")
        except AssertionError:
            logging.error(
                f"Failed test_add_borders: Expected {self.num_borders}, got {len(borders)}"
            )
            raise

    def test_connect_intersections(self):
        intersections = [
            node for node in self.graph.nodes if node.startswith("intersection")
        ]
        for intersection in intersections:
            neighbors = len(
                [
                    node
                    for node in self.graph.neighbors(intersection)
                    if node.startswith("intersection")
                ]
            )
            logging.info(
                f"Intersection {intersection} connected intersections: {neighbors}"
            )
            try:
                self.assertGreaterEqual(neighbors, 2)
                self.assertLessEqual(neighbors, 4)
            except AssertionError:
                logging.error(
                    f"Failed test_connect_intersections: Intersection {intersection} is connected to {neighbors} intersections"
                )
                raise
        logging.info("Passed test_connect_intersections")

    def test_connect_borders(self):
        borders = [node for node in self.graph.nodes if node.startswith("border")]
        for border in borders:
            logging.info(f"Border {border} connections: {self.graph.degree(border)}")
            try:
                self.assertEqual(self.graph.degree(border), 1)
            except AssertionError:
                logging.error(
                    f"Failed test_connect_borders: Border {border} has {self.graph.degree(border)} connections"
                )
                raise
        logging.info("Passed test_connect_borders")

    def test_place_agent(self):
        agent_id = 1
        start_node = self.graph.place_agent(agent_id)
        logging.info(f"Agent {agent_id} placed at {start_node}")
        try:
            self.assertIn(agent_id, self.graph.agent_positions[start_node])
            self.assertTrue(start_node.startswith("border"))
            logging.info("Passed test_place_agent")
        except AssertionError:
            logging.error(
                f"Failed test_place_agent: Agent {agent_id} not placed correctly at {start_node}"
            )
            raise

    def test_move_agent(self):
        agent_id = 1
        start_node = self.graph.place_agent(agent_id)
        new_position = "intersection_0"
        self.graph.move_agent(agent_id, new_position)
        logging.info(f"Agent {agent_id} moved from {start_node} to {new_position}")
        try:
            self.assertEqual(self.graph.agent_positions[agent_id], new_position)
            logging.info("Passed test_move_agent")
        except AssertionError:
            logging.error(
                f"Failed test_move_agent: Agent {agent_id} not moved correctly to {new_position}"
            )
            raise

    def test_save(self):
        filename = "test_graph.pickle"
        self.graph.save(filename)
        logging.info(f"Graph saved to {filename}")
        try:
            self.assertTrue(os.path.exists(filename))
            os.remove(filename)
            logging.info(f"Graph file {filename} removed")
            logging.info("Passed test_save")
        except AssertionError:
            logging.error(f"Failed test_save: File {filename} does not exist")
            raise


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestGraph("test_add_intersections"))
    suite.addTest(TestGraph("test_add_borders"))
    suite.addTest(TestGraph("test_connect_intersections"))
    suite.addTest(TestGraph("test_connect_borders"))
    suite.addTest(TestGraph("test_place_agent"))
    suite.addTest(TestGraph("test_move_agent"))
    suite.addTest(TestGraph("test_save"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
