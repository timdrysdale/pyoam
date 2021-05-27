#!/usr/bin/env python3
import matplotlib.pyplot as plt
from pyoam.demo import two_element

two_element.two_element_el()
plt.savefig("../img/two_element_el.png", dpi=75)

two_element.two_element_az()
plt.savefig("../img/two_element_az.png", dpi=75)
