"""FieldPoint is a ComplexPoint with a zero initial value

The complex value is expected to use to store the magnitide and phase of an
electric field, which is not known at the time the FieldPoint is created.

    Typical Usage:
        x = 1
        y = 2
        z = 3
        f1 = FieldPoint(x,y,z)
        f1.add(1+2j)
        assert(f1.real()==1)

"""
from core.complex_point import ComplexPoint


class FieldPoint(ComplexPoint): # pylint: disable=too-few-public-methods
    """FieldPoints are initialised to zero value because field values
       typically not known at time of setting up calculation"""
    def __init__(self, x, y, z):
        ComplexPoint.__init__(self, x, y, z, 0)
