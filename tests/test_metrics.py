#!/usr/bin/python

from unittest import TestCase, TestSuite, defaultTestLoader
from numpy import array

from tests.context import rr
from tests.context import rr_converter_vector


class MockConfig:
    def __init__(self):
        self.coordinate_converter = rr_converter_vector.NdArrayConverter(array([1, 2]))


class TestMetricSkeleton(TestCase):
    name = 'unspecified'

    @classmethod
    def init(cls, metric_class, distance):
        cls.name = metric_class.__name__
        cls.metric_class = metric_class
        cls.distance = distance
        cls.a = array([-1.0, 5.5])
        cls.b = array([2.0, 1.5])
        cls.c = array([2.0, 4.0])
        cls.weights = array([2.6, 0.65])
        cls.bounds = array([[10, 20], [110, 150]])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = None

    def setUp(self):
        config = MockConfig()
        self.metric.init_weights(config)

    def test_identical(self):
        res = self.metric.calc(self.a, self.a)
        exp = 0.0
        self.assertEqual(exp, res, "{}: identity not given: {} instead of {}".format(self.name, res, exp))

    def test_symmetry(self):
        res = self.metric.calc(self.b, self.a)
        exp = self.metric.calc(self.a, self.b)
        self.assertEqual(exp, res, "{}: Unfulfilled symmetry: {} instead of {}".format(self.name, res, exp))

    def test_triangle_inequality_1(self):
        self._calc_triangle(self.a, self.b, self.c)

    def test_triangle_inequality_2(self):
        self._calc_triangle(self.b, self.c, self.a)

    def test_triangle_inequality_3(self):
        self._calc_triangle(self.c, self.a, self.b)

    def _calc_triangle(self, a, b, c):
        shorter = self.metric.calc(a, c)
        longer = self.metric.calc(a, b) + self.metric.calc(b, c)
        self.assertGreaterEqual(longer, shorter, "{}: Unfulfilled triangle inequality: {} is bigger than {}".format(
            self.name, shorter, longer))

    def test_distance(self):
        res = self.metric.calc(self.a, self.b)
        exp = self.distance
        self.assertAlmostEqual(exp, res, msg="{}: Wrong normal distance: {} instead of {}".format(self.name, res, exp))

    def test_inverted(self):
        res = self.metric.calc(-self.a, -self.b)
        exp = self.distance
        self.assertAlmostEqual(res, exp, msg="{}: Wrong inverted normal distance: {} instead of {}".format(
            self.name, res, exp))


class MetricTestSuite(TestSuite):
    def __init__(self, single_test=None):
        TestSuite.__init__(self)

        class P4Metric(rr.PMetric):
            def __init__(self, *args, **kwargs):
                super().__init__(4, *args, **kwargs)

        metric_tests = [single_test] if single_test is not None else [
            {'metric_class': rr.SumMetric, 'unweighted_distance': 7, 'weighted_distance': 16},
            {'metric_class': rr.EuclidianMetric, 'unweighted_distance': 5, 'weighted_distance': 12.649110640673518},
            {'metric_class': rr.CubicMetric, 'unweighted_distance': 4.497941445, 'weighted_distance': 12.1463558875027},
            {'metric_class': P4Metric, 'unweighted_distance': 4.28457229495381, 'weighted_distance': 12.0368667937382},
            {'metric_class': rr.MaxMetric, 'unweighted_distance': 4, 'weighted_distance': 12},
        ]

        for test in metric_tests:
            self.add_test(test)

    def add_test(self, test):
        class TestUnweightedMetric(TestMetricSkeleton):
            @classmethod
            def init(cls, metric_class, distance):
                super().init(metric_class, distance)
                cls.name += " unweighted"

            def setUp(self):
                self.metric = self.metric_class()
                super().setUp()

        class TestWeightedMetric(TestMetricSkeleton):
            @classmethod
            def init(cls, metric_class, distance):
                super().init(metric_class, distance)
                cls.name += " weighted"

            def setUp(self):
                self.metric = self.metric_class(self.weights)
                super().setUp()

        class TestBoundedMetric(TestMetricSkeleton):
            @classmethod
            def init(cls, metric_class, distance):
                super().init(metric_class, distance)
                cls.name += " bounded"

            def setUp(self):
                self.metric = self.metric_class(self.bounds)
                super().setUp()

        TestUnweightedMetric.init(test['metric_class'], test['unweighted_distance'])
        self.addTest(defaultTestLoader.loadTestsFromTestCase(TestUnweightedMetric))
        TestWeightedMetric.init(test['metric_class'], test['weighted_distance'])
        self.addTest(defaultTestLoader.loadTestsFromTestCase(TestWeightedMetric))
        TestBoundedMetric.init(test['metric_class'], test['weighted_distance'])
        self.addTest(defaultTestLoader.loadTestsFromTestCase(TestBoundedMetric))
