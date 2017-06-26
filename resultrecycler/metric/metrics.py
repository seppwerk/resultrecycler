#!/usr/bin/python

from numpy import zeros, array
from numpy.core.umath import cbrt
from numpy.ma import sqrt
from converter.typecheck import TypeCheck


class Metric:
    # TODO Weigths and Bounds dependend on config
    def __init__(self, weights_or_bounds=None):
        self._raw_weights = weights_or_bounds
        self.weights = None

    def init_weights(self, config):
        if self._raw_weights is None:
            self.weights = 1
        else:
            if TypeCheck.is_scalar(self._raw_weights[config.coordinate_converter.keys[0]]):
                self.weights = config.coordinate_converter.read(self._raw_weights)
            else:
                self.weights = array([1.0 / (self._raw_weights[key][1] - self._raw_weights[key][0])
                                      for key in config.coordinate_converter.keys])
            self.weights /= min(self.weights)

    def calc_distances(self, new_sample, results):
        for res in results:
            res.current_distance = self.calc(res.coordinates, new_sample.coordinates)
        results.sort()

    def calc(self, a, b):
        raise NotImplementedError('Calculation of metric needs to be specified')


class PMetric(Metric):
    def __init__(self, norm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm
        self.inv_norm = 1.0 / self.norm

    def calc(self, a, b):
        return sum((self.weights * abs(a - b)) ** self.norm) ** self.inv_norm


class SumMetric(PMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

    def calc(self, a, b):
        return sum(self.weights * abs(a - b))


class EuclidianMetric(PMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(2, *args, **kwargs)

    def calc(self, a, b):
        return sqrt(sum((self.weights * (a - b))**2))


class CubicMetric(PMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(3, *args, **kwargs)

    def calc(self, a, b):
        return cbrt(sum((self.weights * abs(a - b))**3))


class MaxMetric(Metric):
    def calc(self, a, b):
        return max(self.weights * abs(a - b))
