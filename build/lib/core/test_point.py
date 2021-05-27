"""Test point.py."""

import unittest

from core.point import Point


class TestPoint(unittest.TestCase):
    """Test the initialisation and distance calculation"""
    def test_init(self):
        """Position values match parameters passed to init"""
        x = 1.1
        y = 2.2
        z = 3.3
        p = Point(x, y, z)
        self.assertEqual(p.x, x)
        self.assertEqual(p.y, y)
        self.assertEqual(p.z, z)

    def test_distance_one_axis(self):
        """Distance returns value equal to difference between x-values"""
        p1 = Point(0, 0, 0)
        p2 = Point(1, 0, 0)
        self.assertEqual(p1.distance(p2), 1)
        self.assertEqual(p2.distance(p1), 1)

    def test_distance_two_axis(self):
        """
        Distance returns value equal to diagonal distance
        between point (0,0) and (1,1) in x-y plane
        """
        p1 = Point(0, 0, 0)
        p2 = Point(1, 1, 0)
        self.assertEqual(p1.distance(p2), 2**0.5)
        self.assertEqual(p2.distance(p1), 2**0.5)

    def test_distance_three_axis(self):
        """
        Distance returns value equal to diagonal distance
        between point (0,0,0) and (1,1,1) in 3-D
        """
        p1 = Point(0, 0, 0)
        p2 = Point(1, 1, 1)
        self.assertEqual(p1.distance(p2), 3**0.5)
        self.assertEqual(p2.distance(p1), 3**0.5)

    def test_distance_non_zero_point(self):
        """
        Distance returns value equal to diagonal distance
        between two points, neither of which has a zero
        value component i.e. check all components included
        in calcs
        """
        p1 = Point(-1, -1, -1)
        p2 = Point(1, 1, 1)
        self.assertEqual(p1.distance(p2), 2 * 3**0.5)
        self.assertEqual(p2.distance(p1), 2 * 3**0.5)


if __name__ == "__main__":
    unittest.main()
