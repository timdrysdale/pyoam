"""Test Propagate."""

import unittest

from field_point import FieldPoint
from source_point import SourcePoint
from propagate import propagate_single, propagate
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos
from scipy.constants import speed_of_light


class TestPropagateSingle(unittest.TestCase):
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
    def test_two_element_el(self):
        phis = np.linspace(0, 2 * pi, 100)
        fields = []
        expected = []
        r = 10
        z = 0
        for phi in phis:
            x = r * cos(phi)
            y = r * sin(phi)
            fields.append(FieldPoint(x, y, z))
            expected.append(cos(pi / 2 * cos(phi)))
        frequency = 1e9
        wavelength = speed_of_light / frequency
        x0 = -wavelength / 4
        x1 = +wavelength / 4
        s0 = SourcePoint(x0, 0, 0, 1, 0)
        s1 = SourcePoint(x1, 0, 0, 1, 0)
        sources = [s0, s1]
        propagate(sources, fields, frequency)
        ff = []
        for field in fields:
            ff.append(field.abs())
        ffa = np.array(ff)
        normalised_ffa = ffa / np.max(ffa)
        epsilon = 0.0005
        epsilon_alt = 0.01
        for actual, wanted in zip(normalised_ffa, expected):
            if (wanted > 0.1):  #errors near zero
                self.assertTrue((actual - wanted) < epsilon)
            else:  #errors near zero are bigger
                self.assertTrue((actual - wanted) < epsilon_alt)
        plt.figure()
        plt.plot(phis, ffa / np.max(ffa), label='actual')
        plt.plot(phis, expected, '+', label='wanted')
        plt.xlabel('Theta (radians)')
        plt.ylabel('Normalised amplitude')
        plt.title(
            'Two element pattern with half wavelength spacing - Elevation')
        plt.legend()
        plt.savefig('../../img/TestPropagateTwoElementPatternElevation.png',
                    dpi=150)

    def test_two_element_az(self):
        phis = np.linspace(0, 2 * pi, 100)
        fields = []
        expected = []
        r = 10
        x = 0
        for phi in phis:
            z = r * cos(phi)
            y = r * sin(phi)
            fields.append(FieldPoint(x, y, z))
            expected.append(1)
        frequency = 1e9
        wavelength = speed_of_light / frequency
        x0 = -wavelength / 4
        x1 = +wavelength / 4
        s0 = SourcePoint(x0, 0, 0, 1, 0)
        s1 = SourcePoint(x1, 0, 0, 1, 0)
        sources = [s0, s1]
        propagate(sources, fields, frequency)
        ff = []
        for field in fields:
            ff.append(field.abs())
        ffa = np.array(ff)
        normalised_ffa = ffa / np.max(ffa)
        epsilon = 0.0005
        epsilon_alt = 0.01
        for actual, wanted in zip(normalised_ffa, expected):
            if (wanted > 0.1):  #errors near zero
                self.assertTrue((actual - wanted) < epsilon)
            else:  #errors near zero are bigger
                self.assertTrue((actual - wanted) < epsilon_alt)
        plt.figure()
        plt.plot(phis, ffa / np.max(ffa), label='actual')
        plt.plot(phis, expected, '+', label='wanted')
        plt.xlabel('Phi (radians)')
        plt.ylabel('Normalised amplitude')
        plt.title('Two element pattern with half wavelength spacing - Azimuth')
        plt.legend()
        plt.savefig('../../img/TestPropagateTwoElementPatternAzimuth.png',
                    dpi=150)


if __name__ == "__main__":
    unittest.main()
