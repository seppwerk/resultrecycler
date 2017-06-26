#!/usr/bin/python

from numpy import dot

from resultrecycler.approach.basic import Approach
from config import Information


class FirstDerivative(Approach):
    def __init__(self, config):
        super().__init__(config)
        self.required_information = Information.FirstDerivative

    def guess(self, data, candidate):
        pivot = data[0]
        delta = candidate.coordinates - pivot.coordinates
        return pivot.values + dot(pivot.jacobian, delta)


class SecondDerivative(FirstDerivative):
    def __init__(self, config):
        super().__init__(config)
        self.required_information = Information.SecondDerivative

    def guess(self, data, candidate):
        pivot = data[0]
        delta = candidate.coordinates - pivot.coordinates
        return pivot.values + dot(pivot.jacobian, delta) + dot(dot(delta, pivot.hessian), delta) * 0.5
