"""Demonstrate two-element antenna array calculation."""

from math import pi, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light
from core.field_point import FieldPoint
from core.source_point import SourcePoint
from core.propagate import propagate



def get_sources_two_element():
    """Two sources, half-wavelength spacing, in phase"""
    frequency = 1e9
    wavelength = speed_of_light / frequency
    x0 = -wavelength / 4
    x1 = +wavelength / 4
    s0 = SourcePoint(x0, 0, 0, 1, 0)
    s1 = SourcePoint(x1, 0, 0, 1, 0)
    sources = [s0, s1]
    return sources, frequency

def two_element_el():
    """Two element array, elevation farfield"""
    phis = np.linspace(0, 2 * pi, 100)
    fields = []
    expected = []
    r = 10
    for phi in phis:
        x = r * cos(phi)
        y = r * sin(phi)
        fields.append(FieldPoint(x, y, 0))
        # Expected result from J.D. Kraus Antennas For All Applications
        # 3rd Edition (International) pp.90ff
        expected.append(cos(pi / 2 * cos(phi)))
    sources, frequency = get_sources_two_element()
    propagate(sources, fields, frequency)
    ff = []
    for field in fields:
        ff.append(field.abs())
    ffa = np.array(ff)
    normalised_ffa = ffa / np.max(ffa)
    return phis, normalised_ffa, expected


def two_element_az():
    """Two element array, azimuth farfield"""
    phis = np.linspace(0, 2 * pi, 100)
    fields = []
    expected = []
    r = 10
    for phi in phis:
        y = r * cos(phi)
        z = r * sin(phi)
        fields.append(FieldPoint(0, y, z))
        # Expected result from J.D. Kraus Antennas For All Applications
        # 3rd Edition (International) pp.90ff
        expected.append(1)
    sources, frequency = get_sources_two_element()
    propagate(sources, fields, frequency)
    ff = []
    for field in fields:
        ff.append(field.abs())
    ffa = np.array(ff)
    normalised_ffa = ffa / np.max(ffa)
    return phis, normalised_ffa, expected

def plot_two_element_el():
    # pylint: disable=too-many-locals, duplicate-code
    """Plot the far-field of two element array (elevation)"""
    phis, normalised_ffa, expected = two_element_el()
    plt.figure()
    plt.plot(phis, normalised_ffa, label='actual')
    plt.plot(phis, expected, '+', label='wanted')
    plt.xlabel('Theta (radians)')
    plt.ylabel('Normalised amplitude')
    plt.title(
        'Two element pattern with half wavelength spacing - Elevation')
    plt.legend()

def plot_two_element_az():
    # pylint: disable=too-many-locals, duplicate-code
    """Plot the far-field of two element array (azimuth)"""
    phis, normalised_ffa, expected = two_element_az()
    plt.figure()
    plt.plot(phis, normalised_ffa, label='actual')
    plt.plot(phis, expected, '+', label='wanted')
    plt.xlabel('Phi (radians)')
    plt.ylabel('Normalised amplitude')
    plt.title('Two element pattern with half wavelength spacing - Azimuth')
    plt.legend()



if __name__ == "__main__":
    plot_two_element_el()
    plot_two_element_az()
