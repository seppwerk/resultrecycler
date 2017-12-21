#!/usr/bin/python

from unittest import TestCase, TestSuite, defaultTestLoader
from numpy import array, testing

from tests.context import rr
from tests.context import rr_approach
from tests.context import rr_config


class TestApproach(TestCase):
    @classmethod
    def init_data(cls):
        scalar_sample_data_base = [rr.SampleData(1, 2, 3, 4)]
        scalar_sample_data_ext = [rr.SampleData(1.4, 2.2, 3.6, 4.8)]
        cls.scalar_config = rr_config.Config(scalar_sample_data_base[0])
        cls.scalar_data_1 = [cls.scalar_config.data_class(raw) for raw in scalar_sample_data_base]
        cls.scalar_data_2 = [cls.scalar_config.data_class(raw) for raw in scalar_sample_data_ext]
        cls.scalar_data_2.extend(cls.scalar_data_1)

        cls.scalar_candidate = cls.scalar_config.candidate_class(1.5)
        metric = rr.EuclidianMetric()
        metric.init_weights(cls.scalar_config)
        metric.calc_distances(cls.scalar_candidate, cls.scalar_data_1)
        metric.calc_distances(cls.scalar_candidate, cls.scalar_data_2)

        vector_sample_data_base = [rr.SampleData([1, 1], [2, 2],
                                           [[2, 3], [4, 5]],
                                           [[[4, 5], [5, 6]], [[6, 7], [7, 8]]])]
        vector_sample_data_ext = [rr.SampleData([1.25, 2.25], [3, -1],
                                          [[2.2, 3.4], [-1.5, -2.1]],
                                          [[[0.3, -1.1], [-1.1, 2.4]], [[5, 3], [3, 2]]])]
        cls.vector_config = rr_config.Config(vector_sample_data_base[0])
        cls.vector_data_1 = [cls.vector_config.data_class(raw) for raw in vector_sample_data_base]
        cls.vector_data_2 = [cls.vector_config.data_class(raw) for raw in vector_sample_data_ext]
        cls.vector_data_2.extend(cls.vector_data_1)
        cls.vector_candidate = cls.vector_config.candidate_class([1.5, 2])
        metric = rr.EuclidianMetric()
        metric.init_weights(cls.vector_config)
        metric.calc_distances(cls.vector_candidate, cls.vector_data_1)
        metric.calc_distances(cls.vector_candidate, cls.vector_data_2)

    @classmethod
    def init(cls, approach_class, scalar_result_1, scalar_result_2, vector_result_1, vector_result_2):
        cls.init_data()
        cls.name = approach_class.__name__
        cls.approach_class = approach_class
        cls.scalar_result_1 = scalar_result_1
        cls.scalar_result_2 = scalar_result_2
        cls.vector_result_1 = vector_result_1
        cls.vector_result_2 = vector_result_2

    def setUp(self):
        self.scalar_approach = self.approach_class()
        self.vector_approach = self.approach_class()

    def _evaluate_result(self, approach, data, candidate, exp):
        res = approach.guess(data, candidate)
        testing.assert_array_almost_equal(res, exp, 6,
                                          "{}: Wrong result: {} instead of {}".format(self.name, res, exp))

    def test_correct_scalar_result_1(self):
        self._evaluate_result(self.scalar_approach, self.scalar_data_1, self.scalar_candidate, self.scalar_result_1)

    def test_correct_scalar_result_2(self):
        self._evaluate_result(self.scalar_approach, self.scalar_data_2, self.scalar_candidate, self.scalar_result_2)

    def test_correct_vector_result_1(self):
        self._evaluate_result(self.vector_approach, self.vector_data_1, self.vector_candidate, self.vector_result_1)

    def test_correct_vector_result_2(self):
        self._evaluate_result(self.vector_approach, self.vector_data_2, self.vector_candidate, self.vector_result_2)

    def test_correct_scalar_dim(self):
        result = len(self.scalar_approach.guess(self.scalar_data_2, self.scalar_candidate))
        expected = self.scalar_config.value_converter.dim
        self.assertEqual(result, expected, 'wrong dimension: {} instead of {}'.format(result, expected))

    def test_correct_vector_dim(self):
        result = len(self.vector_approach.guess(self.vector_data_2, self.vector_candidate))
        expected = self.vector_config.value_converter.dim
        self.assertEqual(result, expected, 'wrong dimension: {} instead of {}'.format(result, expected))

    def test_correct_scalar_type(self):
        result = type(self.scalar_approach.guess(self.scalar_data_2, self.scalar_candidate))
        expected = type(array([0]))
        self.assertEqual(result, expected, 'wrong type: {} instead of {}'.format(result, expected))

    def test_correct_vector_type(self):
        result = type(self.vector_approach.guess(self.vector_data_2, self.vector_candidate))
        expected = type(array([0]))
        self.assertEqual(result, expected, 'wrong type: {} instead of {}'.format(result, expected))


class ApproachTestSuite(TestSuite):
    def __init__(self, single_test=None):
        TestSuite.__init__(self)

        class CurrentFixed(rr_approach.Fixed):
            fixed_value = 42

        # TODO one more 2D-test for meaningful affine hull would be nice
        approach_tests = [single_test] if single_test is not None else [
            {'approach_class': CurrentFixed, 'scalar_result_1': array([42]), 'scalar_result_2': array([42]),
             'vector_result_1': array([42, 42]), 'vector_result_2': array([42, 42])},
            {'approach_class': rr.NearestNeighbor, 'scalar_result_1': array([2]), 'scalar_result_2': array([2.2]),
             'vector_result_1': array([2, 2]), 'vector_result_2': array([3, -1])},
            {'approach_class': rr.FirstDerivative, 'scalar_result_1': array([3.5]), 'scalar_result_2': array([2.56]),
             'vector_result_1': array([6, 9]), 'vector_result_2': array([2.7, -0.85])},
            {'approach_class': rr.SecondDerivative, 'scalar_result_1': array([4]), 'scalar_result_2': array([2.584]),
             'vector_result_1': array([12, 17.25]), 'vector_result_2': array([2.853125, -0.81875])},
            {'approach_class': rr.AffineHull, 'scalar_result_1': array([2]), 'scalar_result_2': array([2.25]),
             'vector_result_1': array([2, 2]), 'vector_result_2': array([3, -1])},
            {'approach_class': rr.WeightedSumApproach, 'scalar_result_1': array([2]), 'scalar_result_2': array([13/6]),
             'vector_result_1': array([2, 2]), 'vector_result_2': array([2.7597469, -0.2792408])},
        ]

        for test in approach_tests:
            self.add_test(test)

    def add_test(self, test):
        class CurrentTestApproach(TestApproach):
            pass

        CurrentTestApproach.init(**test)
        self.addTest(defaultTestLoader.loadTestsFromTestCase(CurrentTestApproach))
