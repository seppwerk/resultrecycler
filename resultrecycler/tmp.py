#!/usr/bin/python

from numpy import array, zeros
from numpy.ma import sqrt


a = array([-1, 6])
b = array([2, 2])
w = 1

x = max(w*abs(a-b))
print(x)
