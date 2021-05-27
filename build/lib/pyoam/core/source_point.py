"""SourcePoint is a ComplexPoint with a known initial value

The complex value is expected to use to store the magnitide and phase of a
source defined in terms of its magnitude and phase

    Typical Usage:
        import math
        x = 1
        y = 2
        z = 3
        mag = 1
        phase = math.pi
        s1 = SourcePoint(x,y,z,mag,phase)
        assert(f1.real()==-1)

"""
from cmath import rect
from core.complex_point import ComplexPoint

class SourcePoint(ComplexPoint): # pylint: disable=too-few-public-methods
    """
    SourcePoint is a ComplexPoint whose value is initialised by
    specifying the magnitude and phase
    """
    def __init__(self, x, y, z, mag, phase):# pylint: disable=too-many-arguments
        v = rect(mag, phase)
        ComplexPoint.__init__(self, x, y, z, v)
