"""Test Propagate."""

import unittest

from field_point import FieldPoint
from source_point import SourcePoint
from propagate import propagate_single

from math import pi, atan
from scipy.constants import speed_of_light

class TestPropagateSingle(unittest.TestCase):
    
    def test_decay_rate(self):
        """ check amplitude decays at rate 1/r"""
        s = SourcePoint(0,0,0,1,0)
        f = FieldPoint(1,0,0)
        frequency = 1
        wavelength = speed_of_light / frequency        
        k = 2 * pi / wavelength
        propagate_single(s,f,k)
        self.assertEqual(round(f.real(),6),0.079577)
        self.assertEqual(round(f.imag()/1e-9,6),1.66782)
        
        f2 = FieldPoint(2,0,0)
        propagate_single(s,f2,k)
        self.assertEqual(f.abs()/f2.abs(),2)
        
        f4 = FieldPoint(4,0,0)
        propagate_single(s,f4,k)
        self.assertEqual(round(f.abs()/f4.abs(),6),4)     
        
    def test_phase(self):
        """check phase is 2pi/wavelength"""
        s = SourcePoint(0,0,0,1,0)
        frequency = 1
        wavelength = speed_of_light / frequency        
        k = 2 * pi / wavelength
        f = FieldPoint(wavelength/2,0,0)
        propagate_single(s,f,k)
        self.assertEqual(round(f.phase(),6),3.141593)
        
        # positive phase expected in negative directions because
        # wave travels outward        
        fnegx = FieldPoint(-wavelength/2,0,0)
        propagate_single(s,fnegx,k)
        self.assertEqual(round(fnegx.phase(),6),3.141593) 
        
        f1 = FieldPoint(wavelength,0,0)
        propagate_single(s,f1,k)
        self.assertEqual(round(f1.phase(),6),0)
        
    def test_zero_source(self):
        """check no field if zero mag source"""
        s = SourcePoint(0,0,0,0,pi)
        f = FieldPoint(1,0,0)
        frequency = 1
        wavelength = speed_of_light / frequency        
        k = 2 * pi / wavelength
        propagate_single(s,f,k)
        self.assertEqual(round(f.real(),6),0)
        self.assertEqual(round(f.imag()/1e-9,6),0)

      
if __name__ == "__main__":
    unittest.main()