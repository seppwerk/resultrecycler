#!/usr/bin/python
"""
This file allows to test if a developed metric fulfills the usual requirements of a metric

To run the tests, `dev_setting` needs to be filled with proper values:
    `metric_class`:        The metric class to be tested
    `unweighted distance`: The expected distance between (-1.0, 5.5) and (2.0 and 1.5)
    `weighted distance`:   The expected distance between (-1.0, 5.5) and (2.0 and 1.5) with weights of 4 and 1
"""

import unittest

from numpy import array_equal

from metrics import Metric
from tests.test_metrics import MetricTestSuite


class DiscreteMetric(Metric):
    def calc(self, a, b):
        return 0 if array_equal(a, b) else 1


dev_setting = {
    'metric_class': DiscreteMetric,
    'unweighted_distance': 1,
    'weighted_distance': 1}

if __name__ == '__main__':
    unittest.TextTestRunner().run(MetricTestSuite(dev_setting))