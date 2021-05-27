"""ComplexPoint is a point in space with complex value

The complex value is expected to use to store the magnitide and phase of an 
electric field, either the source or the far field. For efficiency, when 
multiple fields are added using the add() method, there is no pre-calculation
of the different forms of the complex number. These are only calculated on 
demand, by using methods, because this is needed less frequently than the add
method is called.

    Typical Usage:
        x = 1
        y = 2
        z = 3
        v = 1 + 2j
        cp = ComplexPoint(x,y,z,v)
        assert(cp.v==v)
        assert(cp.real() == 1)
        assert(cp.imag() == 2)
        assert(cp.abs() == (1**2+2**2)**0.5)
        import math
        assert(cp.phase() == math.atan(2./1.))
        cp.add(v)
        assert(cp.real() == 2)
        assert(cp.imag() == 4)
        cp.zero()
        assert(cp.real() == 0)
        assert(cp.imag() == 0)       
        

"""
from cmath import phase
from point import Point


class ComplexPoint(Point):
    def __init__(self, x, y, z, value):
        Point.__init__(self, x, y, z)
        self.v = value

    def real(self):
        return self.v.real

    def imag(self):
        return self.v.imag

    def abs(self):
        return abs(self.v)

    def phase(self):
        return phase(self.v)

    def add(self, value):
        self.v = self.v + value

    def zero(self):
        self.v = 0
