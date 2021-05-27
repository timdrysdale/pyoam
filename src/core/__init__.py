"""Core module

Contains the data structure for source and field points
and the functions for propagating sources to field points

SourcePoint and FieldPoint both inherit from ComplexPoint,
and differ only by the initialisation arguments.

Source Point is initialised with a position, and a complex
value (the magnitude and phase of the source).

FieldPoint is automatically initialised with only the position,
while the value is set to zero, because usually only its position
is known when setting up the calculation. If it does need to be
initialised with a value, the add() method can be used.
"""

from . import point
from . import complex_point
from . import source_point
from . import field_point
from . import propagate
