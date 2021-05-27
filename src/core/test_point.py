"""Test point.py."""

import unittest

from core.point import Point


class TestPoint(unittest.TestCase):
    def test_init(self):
        x = 1.1
        y = 2.2
        z = 3.3
        p = Point(x, y, z)
        self.assertEqual(p.x, x)
        self.assertEqual(p.y, y)
        self.assertEqual(p.z, z)

    def test_distance_one_axis(self):
        p1 = Point(0, 0, 0)
        p2 = Point(1, 0, 0)
        self.assertEqual(p1.distance(p2), 1)
        self.assertEqual(p2.distance(p1), 1)

    def test_distance_two_axis(self):
        p1 = Point(0, 0, 0)
        p2 = Point(1, 1, 0)
        self.assertEqual(p1.distance(p2), 2**0.5)
        self.assertEqual(p2.distance(p1), 2**0.5)

    def test_distance_three_axis(self):
        p1 = Point(0, 0, 0)
        p2 = Point(1, 1, 1)
        self.assertEqual(p1.distance(p2), 3**0.5)
        self.assertEqual(p2.distance(p1), 3**0.5)

    def test_distance_non_zero_point(self):
        p1 = Point(-1, -1, -1)
        p2 = Point(1, 1, 1)
        self.assertEqual(p1.distance(p2), 2 * 3**0.5)
        self.assertEqual(p2.distance(p1), 2 * 3**0.5)


if __name__ == "__main__":
    unittest.main()
