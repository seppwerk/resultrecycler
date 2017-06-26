#!/usr/bin/python

from unittest import TestCase, TestSuite, defaultTestLoader

from numpy import array, testing

from converter.typecheck import WrongDimensionError
from data import RawData
from resultrecycler.converter.vector import VectorConverter
from resultrecycler.converter.derivative import DerivativeConverter, ScalarScalarConverter, ScalarEnumConverter, \
    ScalarKeyConverter, EnumScalarConverter, EnumEnumConverter, EnumKeyConverter, KeyScalarConverter, \
    KeyEnumConverter, KeyKeyConverter


class JacobianConverterFailTest(TestCase):
    def setUp(self):
        self.raw_data = RawData({'c': 0.1, 'd': 0.2}, {'a': 0.5, 'b': 0.6},
                                        {'c': {'a': 1.5, 'b': 2.5}, 'd': {'a': 3.5, 'b': 4.5}})
        self.coordinate_converter = VectorConverter.select(self.raw_data.coordinates, 'coordinates')
        self.value_converter = VectorConverter.select(self.raw_data.values, 'values')
        self.select_args = (self.raw_data.jacobian, self.coordinate_converter, self.value_converter)

    def test_wrong_dim_c(self):
        # del self.raw_data.jacobian['d']
        self.assertRaises(WrongDimensionError, DerivativeConverter.select, *self.select_args)


class DerivativeConverterTest(TestCase):
    @classmethod
    def init(cls, converter_class, raw_data, results):
        cls.name = converter_class.__name__
        cls.converter_class = converter_class
        cls.raw_data = RawData(*raw_data)
        cls.conv_jac = array(results['conv_jac'])
        cls.shape_jac = results['shape_jac']
        cls.conv_hess = array(results['conv_hess'])
        cls.shape_hess = results['shape_hess']

    def setUp(self):
        self.coordinate_converter = VectorConverter.select(self.raw_data.coordinates, 'coordinates')
        self.value_converter = VectorConverter.select(self.raw_data.values, 'values')
        self.converter = DerivativeConverter.select(self.raw_data.jacobian,
                                                  self.coordinate_converter, self.value_converter)

    def test_correct_converter(self):
        res = type(self.converter)
        exp = self.converter_class
        self.assertEqual(res, exp, '{}: Wrong converter type: {} instead of {}'.format(self.name, res, exp))

    def test_dim_c(self):
        res = self.converter.dim_c
        exp = self.coordinate_converter.dim
        self.assertEqual(res, exp, '{}: Wrong coordinate dimension: {} instead of {}'.format(self.name, res, exp))

    def test_dim_v(self):
        res = self.converter.dim_v
        exp = self.value_converter.dim
        self.assertEqual(res, exp, '{}: Wrong value dimension: {} instead of {}'.format(self.name, res, exp))

    def test_keys_c(self):
        res = self.converter.keys_c
        exp = self.coordinate_converter.keys
        self.assertEqual(res, exp, '{}: Wrong coordinate keys: {} instead of {}'.format(self.name, res, exp))

    def test_keys_v(self):
        res = self.converter.keys_v
        exp = self.value_converter.keys
        self.assertEqual(res, exp, '{}: Wrong value keys: {} instead of {}'.format(self.name, res, exp))

    def test_raw_jac_conversion(self):
        res = self.converter.read_jac(self.raw_data.jacobian)
        exp = self.conv_jac
        testing.assert_array_equal(res, exp, '{}: Wrong jacobian import: {} instead of {}'.format(self.name, res, exp))

    def test_shape_jac(self):
        res = self.converter.read_jac(self.raw_data.jacobian).shape
        exp = self.shape_jac
        self.assertEqual(res, exp, '{}: Wrong jacobian shape: {} instead of {}'.format(self.name, res, exp))

    def test_raw_hess_conversion(self):
        res = self.converter.read_hess(self.raw_data.hessian)
        exp = self.conv_hess
        testing.assert_array_equal(res, exp, '{}: Wrong hessian import: {} instead of {}'.format(self.name, res, exp))

    def test_shape_hess(self):
        res = self.converter.read_hess(self.raw_data.hessian).shape
        exp = self.shape_hess
        self.assertEqual(res, exp, '{}: Wrong hessian shape: {} instead of {}'.format(self.name, res, exp))


class JacobianConverterTestSuite(TestSuite):
    def __init__(self, single_test=None):
        super().__init__()

        if single_test is None:
            pass
            # self.addTest(defaultTestLoader.loadTestsFromTestCase(JacobianConverterFailTest))

        results = {(1, 1): {'conv_jac': [[2.5]],
                            'shape_jac': (1, 1),
                            'conv_hess': [[[4.1]]],
                            'shape_hess': (1, 1, 1)},
                   (1, 2): {'conv_jac': [[2.5, 2.8]],
                            'shape_jac': (1, 2),
                            'conv_hess': [[[4.1, 4.2], [4.2, 4.4]]],
                            'shape_hess': (1, 2, 2)},
                   (2, 1): {'conv_jac': [[2.5], [3.5]],
                            'shape_jac': (2, 1),
                            'conv_hess': [[[4.1]], [[5.6]]],
                            'shape_hess': (2, 1, 1)},
                   (2, 2): {'conv_jac': [[2.5, 2.8], [3.5, 3.7]],
                            'shape_jac': (2, 2),
                            'conv_hess': [[[4.1, 4.2], [4.2, 4.4]], [[5.6, 5.7], [5.7, 5.9]]],
                            'shape_hess': (2, 2, 2)}
        }

        converter_tests = [single_test] if single_test is not None else [
            {'converter_class': ScalarScalarConverter,
             'raw_data': (0.5, 1.5, 2.5, 4.1),
             'results': results[1, 1]},
            {'converter_class': ScalarEnumConverter,
             'raw_data': (0.5, [1.5, 1.8], [2.5, 2.8], [4.1, 4.2, 4.2, 4.4]),
             'results': results[1, 2]},
            {'converter_class': ScalarKeyConverter,
             'raw_data': (0.5, {'c': 1.5, 'd': 1.8}, {'d': 2.8, 'c': 2.5},
                          {('c', 'c'): 4.1, ('c', 'd'): 4.2, ('d', 'c'): 4.2, ('d', 'd'): 4.4}),
             'results': results[1, 2]},

            {'converter_class': EnumScalarConverter,
             'raw_data': ([0.5, 0.8], 1.5, [2.5, 3.5], [4.1, 5.6]),
             'results': results[2, 1]},

            {'converter_class': EnumEnumConverter,
             'raw_data': ([0.5], [1.5], [[2.5]], [[[4.1]]]),
             'results': results[1, 1]},
            {'converter_class': EnumEnumConverter,
             'raw_data': ([0.5], [1.5, 1.8], [[2.5, 2.8]], [[4.1, 4.2], [4.2, 4.4]]),
             'results': results[1, 2]},
            {'converter_class': EnumEnumConverter,
             'raw_data': ([0.5, 0.8], [1.5], [[2.5], [3.5]], [[[4.1]], [[5.6]]]),
             'results': results[2, 1]},
            {'converter_class': EnumEnumConverter,
             'raw_data': ([0.5, 0.8], [1.5, 1.8], [[2.5, 2.8], [3.5, 3.7]],
                          [[[4.1, 4.2], [4.2, 4.4]], [[5.6, 5.7], [5.7, 5.9]]]),
             'results': results[2, 2]},

            {'converter_class': EnumKeyConverter,
             'raw_data': ([0.5], {'c': 1.5}, [{'c': 2.5}], [{('c', 'c'): 4.1}]),
             'results': results[1, 1]},
            {'converter_class': EnumKeyConverter,
             'raw_data': ([0.5], {'c': 1.5, 'd': 1.8}, [{'d': 2.8, 'c': 2.5}],
                          [{('c', 'c'): 4.1, ('c', 'd'): 4.2, ('d', 'c'): 4.2, ('d', 'd'): 4.4}]),
             'results': results[1, 2]},
            {'converter_class': EnumKeyConverter,
             'raw_data': ([0.5, 0.8], {'c': 1.5}, [{'c': 2.5}, {'c': 3.5}], [{('c', 'c'): 4.1}, {('c', 'c'): 5.6}]),
             'results': results[2, 1]},
            {'converter_class': EnumKeyConverter,
             'raw_data': ([0.5, 0.8], {'c': 1.5, 'd': 1.8}, [{'d': 2.8, 'c': 2.5}, {'c': 3.5, 'd': 3.7}],
                          [{('c', 'c'): 4.1, ('c', 'd'): 4.2, ('d', 'c'): 4.2, ('d', 'd'): 4.4},
                           {('c', 'c'): 5.6, ('c', 'd'): 5.7, ('d', 'c'): 5.7, ('d', 'd'): 5.9}]),
             'results': results[2, 2]},

            {'converter_class': KeyScalarConverter,
             'raw_data': ({'a': 0.5, 'b': 0.8}, 1.5, {'a': 2.5, 'b': 3.5},
                          {'a': 4.1, 'b': 5.6}),
             'results': results[2, 1]},

            {'converter_class': KeyEnumConverter,
             'raw_data': ({'a': 0.5}, [1.5], {'a': [2.5]},
                          {'a': [[4.1]]}),
             'results': results[1, 1]},
            {'converter_class': KeyEnumConverter,
             'raw_data': ({'a': 0.5}, [0.5, 0.8], {'a': [2.5, 2.8]},
                          {'a': [[4.1, 4.2], [4.2, 4.4]]}),
             'results': results[1, 2]},
            {'converter_class': KeyEnumConverter,
             'raw_data': ({'a': 0.5, 'b': 0.8}, [1.5], {'b': [3.5], 'a': [2.5]},
                          {'a': 4.1, 'b': 5.6}),
             'results': results[2, 1]},
            {'converter_class': KeyEnumConverter,
             'raw_data': ({'a': 0.5, 'b': 0.8}, [1.5, 1.8], {'b': [3.5, 3.7], 'a': [2.5, 2.8]},
                          {'a': [[4.1, 4.2], [4.2, 4.4]], 'b': [[5.6, 5.7], [5.7, 5.9]]}),
             'results': results[2, 2]},

            {'converter_class': KeyKeyConverter,
             'raw_data': ({'a': 0.5}, {'c': 1.5}, {'a': {'c': 2.5}},
                          {'a': {('c', 'c'): 4.1}}),
             'results': results[1, 1]},
            {'converter_class': KeyKeyConverter,
             'raw_data': ({'a': 0.5}, {'c': 1.5, 'd': 1.8}, {'a': {'d': 2.8, 'c': 2.5}},
                          {'a': {('c', 'd'): 4.2, ('c', 'c'): 4.1, ('d', 'd'): 4.4, ('d', 'c'): 4.2}}),
             'results': results[1, 2]},
            {'converter_class': KeyKeyConverter,
             'raw_data': ({'a': 0.5, 'b': 0.8}, {'c': 1.5}, {'b': {'c': 3.5}, 'a': {'c': 2.5}},
                          {'b': {('c', 'c'): 5.6}, 'a': {('c', 'c'): 4.1}}),
             'results': results[2, 1]},
            {'converter_class': KeyKeyConverter,
             'raw_data': ({'a': 0.5, 'b': 0.8}, {'c': 1.5, 'd': 1.8},
                          {'b': {'d': 3.7, 'c': 3.5}, 'a': {'c': 2.5, 'd': 2.8}},
                          {'b': {('d', 'c'): 5.7, ('d', 'd'): 5.9, ('c', 'd'): 5.7, ('c', 'c'): 5.6},
                           'a': {('c', 'd'): 4.2, ('c', 'c'): 4.1, ('d', 'd'): 4.4, ('d', 'c'): 4.2}}),
             'results': results[2, 2]}
        ]

        for test in converter_tests:
            self.add_test(test)

    def add_test(self, test):
        class CurrentVectorConverterTest(DerivativeConverterTest):
            pass

        CurrentVectorConverterTest.init(**test)
        self.addTest(defaultTestLoader.loadTestsFromTestCase(CurrentVectorConverterTest))
