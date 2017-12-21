#!/usr/bin/python

from unittest import TestCase, TestSuite, defaultTestLoader

from numpy import array, testing
from converter.typecheck import TypeCheck
from data import SampleData

from .context import rr


class ResultRecyclerTest(TestCase):
    @classmethod
    def init(cls, name, datas, candidates, results, raw_limit=None, metric=None, approach_class=None):
        cls.name = name
        cls.datas = datas
        cls.candidates = candidates
        cls.results = results
        cls.raw_limit = raw_limit
        cls.metric = metric
        cls.approach_class = approach_class

    def setUp(self):
        self.rr = rr.ResultRecycler(self.raw_limit, self.metric, self.approach_class)

    def test_correct_results(self):
        for dat, cand, exp in zip(self.datas, self.candidates, self.results):
            if dat is not None:
                self.rr.add_data(dat)
            res = self.rr.calculate(cand)
            if TypeCheck.is_scalar(exp):
                exp = array([exp])
            else:
                exp = array(exp)
            testing.assert_array_almost_equal(res, exp,
                                              err_msg='{}: Wrong result: {} instead of {}'.format(self.name, res, exp))


class ResultRecyclerTestSuite(TestSuite):
    def __init__(self, single_test=None):
        super().__init__()

        converter_tests = [single_test] if single_test is not None else [
            {'name': 'SingleAffine', 'datas': [SampleData(1, 2), SampleData(1.8, 3)],
             'candidates': [2, 2], 'results': [2, 3.25]},
            {'name': 'SingleDer1', 'datas': [SampleData(1, 2, 3), SampleData(1.8, 4, 1)],
             'candidates': [2, 2], 'results': [5, 4.2]},
            {'name': 'SingleDer2', 'datas': [SampleData(1, 2, 3, 4), SampleData(1.8, 4, 1, 2)],
             'candidates': [2, 2], 'results': [7, 4.24]},
            {'name': 'MultiAffine', 'datas': [SampleData([1, 1], [2, 2])],
             'candidates': [[1.5, 1.5]], 'results': [[2, 2]]},
            {'name': 'MultiDer1', 'datas': [SampleData([1, 1], [2, 2], [[2, 2], [2, 2]])],
             'candidates': [[1.5, 1.5]], 'results': [[4, 4]]},
            {'name': 'MultiDer2', 'datas': [SampleData([1, 1], [2, 2], [[2, 2], [2, 2]],
                                                       [[[3, 3], [3, 3]], [[3, 3], [3, 3]]])],
             'candidates': [[1.5, 1.5]], 'results': [[5.5, 5.5]]},
        ]

        for test in converter_tests:
            self.add_test(test)

    def add_test(self, test):
        class CurrentResultRecyclerTest(ResultRecyclerTest):
            pass

        CurrentResultRecyclerTest.init(**test)
        self.addTest(defaultTestLoader.loadTestsFromTestCase(CurrentResultRecyclerTest))
