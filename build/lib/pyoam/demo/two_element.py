"""Demonstrate two-element antenna array calculation."""

from core.field_point import FieldPoint
from core.source_point import SourcePoint
from core.propagate import propagate_single, propagate
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos
from scipy.constants import speed_of_light

    def two_element_el():
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
        plt.savefig('TestPropagateTwoElementPatternElevation.png',
                    dpi=150)

    def two_element_az():
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
        plt.savefig('TestPropagateTwoElementPatternAzimuth.png',
                    dpi=150)


if __name__ == "__main__":
    two_element_el()
    two_element_az()
