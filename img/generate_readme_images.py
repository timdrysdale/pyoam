#!/usr/bin/env python3
from ..src.demo import two_element

two_element.two_element_az()
plt.savefig('TestPropagateTwoElementPatternAzimuth.png',
                    dpi=150)

two_element.two_element_el()
plt.savefig('TestPropagateTwoElementPatternElevation.png',
                    dpi=150)
