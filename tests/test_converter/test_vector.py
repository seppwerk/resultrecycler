#!/usr/bin/python

from unittest import TestCase, TestSuite, defaultTestLoader

from numpy import array, testing

from resultrecycler.converter.vector import ScalarConverter, NdArrayConverter, ListConverter, TupleConverter, \
    DictConverter, UnsupportedTypeError, VectorConverter


class VectorConverterFailTest(TestCase):
    def test_fail(self):
        self.assertRaises(UnsupportedTypeError, VectorConverter.select, {2, 3.5}, 'testing')


class VectorConverterTest(TestCase):
    @classmethod
    def init(cls, converter_class, raw_data, conv_data, dim, keys):
        cls.name = converter_class.__name__
        cls.converter_class = converter_class
        cls.raw_data = raw_data
        cls.conv_data = conv_data
        cls.dim = dim
        cls.keys = keys

    def setUp(self):
        self.converter = VectorConverter.select(self.raw_data, 'testing')

    def test_correct_converter(self):
        res = type(self.converter)
        exp = self.converter_class
        self.assertEqual(res, exp, '{}: Wrong converter: {} instead of {}'.format(self.name, res, exp))

    def test_raw_conversion(self):
        res = self.converter.read(self.raw_data)
        exp = self.conv_data
        testing.assert_array_equal(res, exp, '{}: Wrong raw import: {} instead of {}'.format(self.name, res, exp))

    def test_back_conversion_data(self):
        res = self.converter.export(self.conv_data)
        exp = self.raw_data
        if isinstance(self.raw_data, (int, float)):
            self.assertEqual(res, exp, '{}: Wrong export: {} instead of {}'.format(self.name, res, exp))
        else:
            testing.assert_array_equal(res, exp, '{}: Wrong export: {} instead of {}'.format(self.name, res, exp))

    def test_back_conversion_type(self):
        if isinstance(self.raw_data, (int, float)):
            return
        res = type(self.converter.export(self.conv_data))
        exp = type(self.raw_data)
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
            {'converter_class': ScalarConverter, 'raw_data': 2.5, 'conv_data': array([2.5]),
             'dim': 1, 'keys': (0, )},
            {'converter_class': NdArrayConverter, 'raw_data': array([2, 3.5]), 'conv_data': array([2, 3.5]),
             'dim': 2, 'keys': (0, 1)},
            {'converter_class': ListConverter, 'raw_data': [2, 3.5], 'conv_data': array([2, 3.5]),
             'dim': 2, 'keys': (0, 1)},
            {'converter_class': TupleConverter, 'raw_data': (2, 3.5), 'conv_data': array([2, 3.5]),
             'dim': 2, 'keys': (0, 1)},
            {'converter_class': DictConverter, 'raw_data': {'a': 2, 'b': 3.5}, 'conv_data': array([2, 3.5]),
             'dim': 2, 'keys': ('a', 'b')},
        ]

        for test in converter_tests:
            self.add_test(test)

    def add_test(self, test):
            class CurrentVectorConverterTest(VectorConverterTest):
                pass
            CurrentVectorConverterTest.init(**test)
            self.addTest(defaultTestLoader.loadTestsFromTestCase(CurrentVectorConverterTest))
