"""Test point.py."""

import unittest
import math
from .complex_point import ComplexPoint



class TestComplexPoint(unittest.TestCase):
    """ Test distance and value methods"""
    def test_init(self):
        """Position and value equal to initialisation parameters"""
        x = 1.1
        y = 2.2
        z = 3.3
        v = 1 + 1j
        p = ComplexPoint(x, y, z, v)
        self.assertEqual(p.x, x)
        self.assertEqual(p.y, y)
        self.assertEqual(p.z, z)
        self.assertEqual(p.v, v)

    def test_distance(self):
        """Distance in 3D works as for parent"""
        p1 = ComplexPoint(-1, -1, -1, 0)
        p2 = ComplexPoint(1, 1, 1, 0)
        self.assertEqual(p1.distance(p2), 2 * 3**0.5)
        self.assertEqual(p2.distance(p1), 2 * 3**0.5)

    def test_add_value(self):
        """Add value results in sum of previous and new value"""
        p1 = ComplexPoint(0, 0, 0, 1 + 2j)
        self.assertEqual(p1.real(), 1)
        self.assertEqual(p1.imag(), 2)
        self.assertEqual(p1.abs(), (1**2 + 2 * 2)**0.5)
        self.assertEqual(p1.phase(), math.atan(2))
        p1.add(2 + 2j)
        self.assertEqual(p1.real(), 3)
        self.assertEqual(p1.imag(), 4)
        self.assertEqual(p1.abs(), (3**2 + 4**2)**0.5)
        self.assertEqual(p1.phase(), math.atan(4. / 3.))


if __name__ == "__main__":
    unittest.main()
