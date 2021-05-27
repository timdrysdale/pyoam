"""Test Propagate."""

import unittest
from math import pi
from scipy.constants import speed_of_light
from core.field_point import FieldPoint
from core.source_point import SourcePoint
from core.propagate import propagate_single
from demo.two_element import two_element_el, two_element_az

class TestPropagateSingle(unittest.TestCase):
    """Test single souce point is propagated correctly"""
    def test_decay_rate(self):
        """ check amplitude decays at rate 1/r"""
        s = SourcePoint(0, 0, 0, 1, 0)
        f = FieldPoint(1, 0, 0)
        frequency = 1
        wavelength = speed_of_light / frequency
        k = 2 * pi / wavelength
        propagate_single(s, f, k)
        self.assertEqual(round(f.real(), 6), 0.079577)
        self.assertEqual(round(f.imag() / 1e-9, 6), 1.66782)

        f2 = FieldPoint(2, 0, 0)
        propagate_single(s, f2, k)
        self.assertEqual(f.abs() / f2.abs(), 2)

        f4 = FieldPoint(4, 0, 0)
        propagate_single(s, f4, k)
        self.assertEqual(round(f.abs() / f4.abs(), 6), 4)

    def test_phase(self):
        """check phase is 2pi/wavelength"""
        s = SourcePoint(0, 0, 0, 1, 0)
        frequency = 1
        wavelength = speed_of_light / frequency
        k = 2 * pi / wavelength
        f = FieldPoint(wavelength / 2, 0, 0)
        propagate_single(s, f, k)
        self.assertEqual(round(f.phase(), 6), 3.141593)

        # positive phase expected in negative directions because
        # wave travels outward
        fnegx = FieldPoint(-wavelength / 2, 0, 0)
        propagate_single(s, fnegx, k)
        self.assertEqual(round(fnegx.phase(), 6), 3.141593)

        f1 = FieldPoint(wavelength, 0, 0)
        propagate_single(s, f1, k)
        self.assertEqual(round(f1.phase(), 6), 0)

    def test_zero_source(self):
        """check no field if zero mag source"""
        s = SourcePoint(0, 0, 0, 0, pi)
        f = FieldPoint(1, 0, 0)
        frequency = 1
        wavelength = speed_of_light / frequency
        k = 2 * pi / wavelength
        propagate_single(s, f, k)
        self.assertEqual(round(f.real(), 6), 0)
        self.assertEqual(round(f.imag() / 1e-9, 6), 0)



class TestPropagate(unittest.TestCase):
    """Test propagation from multiple source points"""
    def test_two_element_el(self):
        """Test elevation results for two-element half-wavelength antenna"""
        _, normalised_ffa, expected = two_element_el()
        epsilon = 0.0005
        epsilon_alt = 0.01
        for actual, wanted in zip(normalised_ffa, expected):
            if wanted > 0.1:  #errors near zero
                self.assertTrue((actual - wanted) < epsilon)
            else:  #errors near zero are bigger
                self.assertTrue((actual - wanted) < epsilon_alt)

    def test_two_element_az(self):
        """Test elevation results for two-element half-wavelength antenna"""
        _, normalised_ffa, expected = two_element_az()
        epsilon = 0.0005
        epsilon_alt = 0.01
        for actual, wanted in zip(normalised_ffa, expected):
            if wanted > 0.1:  #errors near zero
                self.assertTrue((actual - wanted) < epsilon)
            else:  #errors near zero are bigger
                self.assertTrue((actual - wanted) < epsilon_alt)

if __name__ == "__main__":
    unittest.main()
