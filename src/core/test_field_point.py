"""Test FieldPoint."""

import unittest
import math
from core.field_point import FieldPoint


class TestFieldPoint(unittest.TestCase):
    """Test distance() and add()"""
    def test_init(self):
        """Position and value match initialisation parameters"""
        x = 1.1
        y = 2.2
        z = 3.3
        fp = FieldPoint(x, y, z)
        self.assertEqual(fp.x, x)
        self.assertEqual(fp.y, y)
        self.assertEqual(fp.z, z)
        self.assertEqual(fp.v, 0)

    def test_distance(self):
        """Distance between two points is correct"""
        fp1 = FieldPoint(-1, -1, -1)
        fp2 = FieldPoint(1, 1, 1)
        self.assertEqual(fp1.distance(fp2), 2 * 3**0.5)
        self.assertEqual(fp2.distance(fp1), 2 * 3**0.5)

    def test_add_value(self):
        """Values are added not replaced"""
        fp1 = FieldPoint(0, 0, 0)
        fp1.add(1 + 2j)
        self.assertEqual(fp1.real(), 1)
        self.assertEqual(fp1.imag(), 2)
        self.assertEqual(fp1.abs(), (1**2 + 2 * 2)**0.5)
        self.assertEqual(fp1.phase(), math.atan(2))
        fp1.add(2 + 2j)
        self.assertEqual(fp1.real(), 3)
        self.assertEqual(fp1.imag(), 4)
        self.assertEqual(fp1.abs(), (3**2 + 4**2)**0.5)
        self.assertEqual(fp1.phase(), math.atan(4. / 3.))


if __name__ == "__main__":
    unittest.main()
