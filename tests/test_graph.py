import os
import unittest
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from graph import Graph


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
        print("Setup complete: Graph initialized")

    def test_add_intersections(self):
        intersections = [
            node for node in self.graph.nodes if node.startswith("intersection")
        ]
        print(f"Intersections added: {intersections}")
        self.assertEqual(len(intersections), self.num_intersections)
        print("Passed test_add_intersections")

    def test_add_borders(self):
        borders = [node for node in self.graph.nodes if node.startswith("border")]
        print(f"Borders added: {borders}")
        self.assertEqual(len(borders), self.num_borders)
        print("Passed test_add_borders")

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
            print(f"Intersection {intersection} connected intersections: {neighbors}")
            self.assertGreaterEqual(neighbors, 2)
            self.assertLessEqual(neighbors, 4)
        print("Passed test_connect_intersections")

    def test_connect_borders(self):
        borders = [node for node in self.graph.nodes if node.startswith("border")]
        for border in borders:
            print(f"Border {border} connections: {self.graph.degree(border)}")
            self.assertEqual(self.graph.degree(border), 1)
        print("Passed test_connect_borders")

    def test_place_agent(self):
        agent_id = 1
        start_node = self.graph.place_agent(agent_id)
        print(f"Agent {agent_id} placed at {start_node}")
        self.assertIn(agent_id, self.graph.agent_positions[start_node])
        self.assertTrue(start_node.startswith("border"))
        print("Passed test_place_agent")

    def test_move_agent(self):
        agent_id = 1
        start_node = self.graph.place_agent(agent_id)
        new_position = "intersection_0"
        self.graph.move_agent(agent_id, new_position)
        print(f"Agent {agent_id} moved from {start_node} to {new_position}")
        self.assertEqual(self.graph.agent_positions[agent_id], new_position)
        print("Passed test_move_agent")

    def test_save(self):
        filename = "test_graph.pickle"
        self.graph.save(filename)
        print(f"Graph saved to {filename}")
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)
        print(f"Graph file {filename} removed")
        print("Passed test_save")


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
