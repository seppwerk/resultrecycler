#!/usr/bin/python

from unittest import TestCase, TestSuite, defaultTestLoader

from numpy import array, testing

from tests.context import rr_converter_vector as rr_vector


class VectorConverterFailTest(TestCase):
    def test_fail(self):
        self.assertRaises(rr_vector.UnsupportedTypeError, rr_vector.VectorConverter.select, {2, 3.5}, 'testing')


class VectorConverterTest(TestCase):
    @classmethod
    def init(cls, converter_class, sample_data, conv_data, dim, keys):
        cls.name = converter_class.__name__
        cls.converter_class = converter_class
        cls.sample_data = sample_data
        cls.conv_data = conv_data
        cls.dim = dim
        cls.keys = keys

    def setUp(self):
        self.converter = rr_vector.VectorConverter.select(self.sample_data, 'testing')

    def test_correct_converter(self):
        res = type(self.converter)
        exp = self.converter_class
        self.assertEqual(res, exp, '{}: Wrong converter: {} instead of {}'.format(self.name, res, exp))

    def test_raw_conversion(self):
        res = self.converter.read(self.sample_data)
        exp = self.conv_data
        testing.assert_array_equal(res, exp, '{}: Wrong raw import: {} instead of {}'.format(self.name, res, exp))

    def test_back_conversion_data(self):
        res = self.converter.export(self.conv_data)
        exp = self.sample_data
        if isinstance(self.sample_data, (int, float)):
            self.assertEqual(res, exp, '{}: Wrong export: {} instead of {}'.format(self.name, res, exp))
        else:
            testing.assert_array_equal(res, exp, '{}: Wrong export: {} instead of {}'.format(self.name, res, exp))

    def test_back_conversion_type(self):
        if isinstance(self.sample_data, (int, float)):
            return
        res = type(self.converter.export(self.conv_data))
        exp = type(self.sample_data)
        self.assertEqual(res, exp, '{}: Wrong export type: {} instead of {}'.format(self.name, res, exp))

    def test_dim(self):
        res = self.converter.dim
        exp = self.dim
        self.assertEqual(res, exp, '{}: Wrong dimension: {} instead of {}'.format(self.name, res, exp))

    def test_keys(self):
        res = self.converter.keys
        exp = self.keys
        self.assertEqual(res, exp, '{}: Wrong keys: {} instead of {}'.format(self.name, res, exp))


class VectorConverterTestSuite(TestSuite):
    def __init__(self, single_test=None):
        super().__init__()

        if single_test is None:
            self.addTest(defaultTestLoader.loadTestsFromTestCase(VectorConverterFailTest))

        converter_tests = [single_test] if single_test is not None else [
            {'converter_class': rr_vector.ScalarConverter, 'sample_data': 2.5, 'conv_data': array([2.5]),
             'dim': 1, 'keys': (0, )},
            {'converter_class': rr_vector.NdArrayConverter, 'sample_data': array([2, 3.5]), 'conv_data': array([2, 3.5]),
             'dim': 2, 'keys': (0, 1)},
            {'converter_class': rr_vector.ListConverter, 'sample_data': [2, 3.5], 'conv_data': array([2, 3.5]),
             'dim': 2, 'keys': (0, 1)},
            {'converter_class': rr_vector.TupleConverter, 'sample_data': (2, 3.5), 'conv_data': array([2, 3.5]),
             'dim': 2, 'keys': (0, 1)},
            {'converter_class': rr_vector.DictConverter, 'sample_data': {'a': 2, 'b': 3.5}, 'conv_data': array([2, 3.5]),
             'dim': 2, 'keys': ('a', 'b')},
        ]

        for test in converter_tests:
            self.add_test(test)

    def add_test(self, test):
            class CurrentVectorConverterTest(VectorConverterTest):
                pass
            CurrentVectorConverterTest.init(**test)
            self.addTest(defaultTestLoader.loadTestsFromTestCase(CurrentVectorConverterTest))
