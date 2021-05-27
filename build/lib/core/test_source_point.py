"""Test SourcePoint."""

import unittest
import math
from .source_point import SourcePoint



class TestSourcePoint(unittest.TestCase):
    """Test distance and add"""
    def test_init(self):
        """Position and value same as initialisation parameters"""
        x = 1.1
        y = 2.2
        z = 3.3
        mag = 1
        phase = math.pi
        p = SourcePoint(x, y, z, mag, phase)
        self.assertEqual(p.x, x)
        self.assertEqual(p.y, y)
        self.assertEqual(p.z, z)
        self.assertEqual(p.abs(), 1)

    def test_distance(self):
        """Distance returns distance between two points"""
        p1 = SourcePoint(-1, -1, -1, 0, 0)
        p2 = SourcePoint(1, 1, 1, 0, 0)
        self.assertEqual(p1.distance(p2), 2 * 3**0.5)
        self.assertEqual(p2.distance(p1), 2 * 3**0.5)

    def test_add_value(self):
        """Add sums previous and new value, not replace"""
        p1 = SourcePoint(0, 0, 0, 1, 0)
        p1.add(1 + 2j)
        self.assertEqual(p1.real(), 2)
        self.assertEqual(p1.imag(), 2)
        self.assertEqual(p1.abs(), (2**2 + 2 * 2)**0.5)
        self.assertEqual(p1.phase(), math.atan(1))


if __name__ == "__main__":
    unittest.main()
