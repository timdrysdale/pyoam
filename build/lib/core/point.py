"""Represent points in three-dimensional Cartesian geometry.

Points in three-dimensional Cartesian geometry have x,y, and z coordinates. The
distance between two points is of interest for calculating field values.

   Typical usage:
       p1 = Point(1.0,0,0)
       p2 = Point(0,0,0)
       d = p1.distance(p2)
       assert(d==1.0)

"""

import math


class Point: # pylint: disable=too-few-public-methods
    """Three dimensional point in Cartesian geometry.

    Attributes:
        x: A float for the x-axis value
        y: A float for the y-axis value
        xz A float for the z-axis value
    """
    def __init__(self, x, y, z):
        """Inits Point with its position."""
        self.x = x
        self.y = y
        self.z = z

    def distance(self, point):
        """Calculates the distance to/from another Point (always positive)"""
        return math.sqrt((self.x - point.x)**2 + (self.y - point.y)**2 +
                         (self.z - point.z)**2)
