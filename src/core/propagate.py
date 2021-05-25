"""Propagate from sources to field points.

A list of SourcePoints represent an antenna array, while a list of FieldPoints 
represents the farfield points of interest. For a given frequency, the 
farfield is calculated by summing the contribution from each source point, at
each point in the farfield. Isotropic radiators are assumed, and the frequency
is given in Hertz, for dimensions in metres.

    Typical usage:
        # uniform linear array of two elements, 10mm apart on x-axis
        import numpy as np
        import matplotlib.pyplot as plt
        x1 = -5e-3
        x2 = 5e-3
        y = 0
        z = 0
        mag = 1
        phase = 0
        s1 = SourcePoint(x1,y,z,mag,phase)
        s2 = SourcePoint(x2,y,z,mag,phase)
        sources = [s1,s2]
        z2 = 0.5
        fields = []
        xff = np.linspace(-0.1,0.1,10)
        for xf in xff:
            fields.append(FieldPoint(xf,y,z2))
        freq = 5e9 #5GHz    
        propagate(sources,fields,freq) 
        plt.figure()
        ff = []
        for p in fields:
            ff.append(p.abs())
        plt.plot(xf,ff)

"""
from scipy.constants import speed_of_light, pi
from cmath import exp

def propagate(sources, fields, frequencyHz):
    wavelength = speed_of_light / frequencyHz
    k = 2. * pi / wavelength
    for m in range(len(fields)):
        for n in range(len(sources)): 
            fields[m] = propagate_single(sources[n],fields[m],k)
            #Check that the field is modified in calling scope
        
def propagate_single(source,field,k):
    r = source.distance(field)
    A = source.v * exp(1j*k*r) / (4 * pi * r)
    field.add(A)
    return field    
