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
from complex_point import ComplexPoint
from cmath import rect

class SourcePoint(ComplexPoint):
    
    def __init__(self, x,y,z, mag, phase):
        v = rect(mag,phase)
        ComplexPoint.__init__(self,x,y,z,v)
