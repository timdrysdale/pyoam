#!/usr/bin/env python3

from math import cos, sin
import matplotlib.pyplot as plt
import numpy as np
from pyoam.core.source_point import SourcePoint
from pyoam.core.field_point import FieldPoint
from pyoam.core.propagate import propagate
from scipy.constants import pi, speed_of_light

# set up the source point (two element antenna)
frequency = 1e9
wavelength = speed_of_light / frequency
x0 = -wavelength / 4
x1 = +wavelength / 4
s0 = SourcePoint(x0, 0, 0, 1, 0)
s1 = SourcePoint(x1, 0, 0, 1, 0)
sources = [s0, s1]

# set up the field points (far field, circle)
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

#do the calculation
propagate(sources, fields, frequency)

#extract the magnitude of the field
ff = []
for field in fields:
    ff.append(field.abs())
ffa = np.array(ff)

#normalise the array factor results so it can be compared with
#normalised result from the text book
normalised_ffa = ffa / np.max(ffa)

# plot the results
plt.figure()
plt.plot(phis, normalised_ffa, label='actual')
plt.plot(phis, expected, '+', label='wanted')
plt.xlabel('Theta (radians)')
plt.ylabel('Normalised amplitude')
plt.title(
    'Two element pattern with half wavelength spacing - Elevation')
plt.legend()
