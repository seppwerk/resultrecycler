#!/usr/bin/python

from numpy import dot

from .basic import Approach
from ..config import Information


class FirstDerivative(Approach):
    """
    Uses a first degree taylor polynomial from the closest sample.
    """
    def __init__(self):
        super().__init__()
        self.required_information = Information.FirstDerivative

    def guess(self, data, candidate):
        pivot = data[0]
        delta = candidate.coordinates - pivot.coordinates
        return pivot.values + dot(pivot.jacobian, delta)


class SecondDerivative(FirstDerivative):
    """
    Uses a second degree taylor polynomial from the closest sample.
    """
    def __init__(self):
        super().__init__()
        self.required_information = Information.SecondDerivative

    def guess(self, data, candidate):
        pivot = data[0]
        delta = candidate.coordinates - pivot.coordinates
        return pivot.values + dot(pivot.jacobian, delta) + dot(dot(pivot.hessian, delta), delta) * 0.5
