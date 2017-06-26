#!/usr/bin/python

from unittest import TestCase, TestSuite, defaultTestLoader

from limits import Limit, RawLimit


class TestLimitImport(TestCase):
    @classmethod
    def init_data(cls):
        cls.minimum = [1, 2.25]
        cls.default = [3, 3.5]
        cls.maximum = [5, 4.75]

        cls.common = [3, 4.4]

        cls.individual = [[5, 5.5], [6.1]]

    @classmethod
    def init(cls, name, raw_limit):
        cls.init_data()
        cls.name = name
        cls.raw_limit = raw_limit

    def setUp(self):
        self.limit = Limit(self.raw_limit)

    def isIn(self, item, container, field):
        self.assertIn(item, container, '{}: Missing {}: {} not in {}'.format(self.name, field, item, container))

    def isAt(self, item, container, position, field):
        self.assertEqual(item, container[position], '{}: Wrong position in {}: {} not at {} in {}'
                         .format(self.name, field, item, position, container))

    def checkSingleValueCorrect(self, item, container, position, field):
        self.isIn(item, container, field)
        self.isAt(item, container, position, field)

    def checkMultiValueCorrect(self, item, container, position, field):
        self.isIn(item, container[position], field)

    def test_min_set(self):
        for key, minimum in enumerate(self.minimum):
            self.checkSingleValueCorrect(minimum, self.limit.minimum, key, 'Minimum')

    def test_max_set(self):
        for key, maximum in enumerate(self.maximum):
            self.checkSingleValueCorrect(maximum, self.limit.maximum, key, 'Maximum')

    def test_default_set(self):
        for key, default in enumerate(self.default):
            self.checkSingleValueCorrect(default, self.limit.default, key, 'Default')

    def test_fixed_set(self):
        for key, fixed in enumerate(self.minimum):
            self.checkMultiValueCorrect(fixed, self.limit.fixed_values, key, 'Min in Fixed')
        for key, fixed in enumerate(self.maximum):
            self.checkMultiValueCorrect(fixed, self.limit.fixed_values, key, 'Max in Fixed')
        for fixed in self.common:
            for key in range(len(self.minimum)):
                self.checkMultiValueCorrect(fixed, self.limit.fixed_values, key, 'Common in Fixed')
        for key, individuals in enumerate(self.individual):
            for fixed in individuals:
                self.checkMultiValueCorrect(fixed, self.limit.fixed_values, key, 'Individual in Fixed')


class TestSameLimitImport(TestLimitImport):
    @classmethod
    def init_data(cls):
        cls.minimum = [2.25, 2.25]
        cls.default = [3.5, 3.5]
        cls.maximum = [4.75, 4.75]

        cls.common = [3, 4.4]

        cls.individual = [[5, 5.5], [5, 5.5]]


class TestScalarLimitImport(TestLimitImport):
    @classmethod
    def init_data(cls):
        cls.minimum = [2.25]
        cls.default = [3.5]
        cls.maximum = [4.75]

        cls.common = [3, 4.4]

        cls.individual = [[5, 5.5]]


class LimitImportTestSuite(TestSuite):
    def __init__(self):
        TestSuite.__init__(self)

        mixed_tests = [
            {'name': 'Enumerable', 'raw_limit': RawLimit([1, 2.25], [5, 4.75], [3, 3.5], [3, 4.4], [[5, 5.5], [6.1]])},
            {'name': 'Dictionary', 'raw_limit': RawLimit({'b': 2.25, 'a': 1}, {'a': 5, 'b': 4.75}, {'b': 3.5, 'a': 3},
                                                         [3, 4.4], {'a': [5, 5.5], 'b': [6.1]})},
            {'name': 'EnumKeys', 'raw_limit': RawLimit([1, 2.25], [5, 4.75], [3, 3.5], [3, 4.4], [[5, 5.5], [6.1]])},
            {'name': 'DictKeys', 'raw_limit': RawLimit({'b': 2.25, 'a': 1}, {'a': 5, 'b': 4.75}, {'b': 3.5, 'a': 3},
                                                       [3, 4.4], {'a': [5, 5.5], 'b': [6.1]}, ['b', 'a'])},
            {'name': 'NoDefault', 'raw_limit': RawLimit([1, 2.25], [5, 4.75], None, [3, 4.4], [[5, 5.5], [6.1]])},
            {'name': 'NoCommon', 'raw_limit': RawLimit([1, 2.25], [5, 4.75], [3, 3.5],
                                                       None, [[3, 4.4, 5, 5.5], [3, 4.4, 6.1]])},
            {'name': 'NoIndividual', 'raw_limit': RawLimit([1, 2.25], [5, 4.75], [3, 3.5],
                                                           [3, 4.4, 5, 5.5, 6.1])},
        ]

        for test in mixed_tests:
            self.add_test(test, TestLimitImport)

        same_tests = [
            {'name': 'EnumBase', 'raw_limit': RawLimit(2.25, 4.75, 3.5, [3, 4.4, 5, 5.5], value_keys=[1, 0])},
            {'name': 'EnumCount', 'raw_limit': RawLimit(2.25, 4.75, 3.5, [3, 4.4, 5, 5.5], value_keys=2)},
            {'name': 'EnumNoDef', 'raw_limit': RawLimit(2.25, 4.75, None, [3, 4.4, 5, 5.5], value_keys=[1, 0])},
            {'name': 'EnumMin', 'raw_limit': RawLimit([2.25, 2.25], 4.75, 3.5, [3, 4.4, 5, 5.5])},
            {'name': 'EnumMax', 'raw_limit': RawLimit(2.25, [4.75, 4.75], 3.5, [3, 4.4, 5, 5.5])},
            {'name': 'EnumDef', 'raw_limit': RawLimit(2.25, 4.75, [3.5, 3.5], [3, 4.4, 5, 5.5])},
            {'name': 'EnumFix', 'raw_limit': RawLimit(2.25, 4.75, 3.5, [3, 4.4], [[5, 5.5], [5, 5.5]])},
            {'name': 'DictBase', 'raw_limit': RawLimit(2.25, 4.75, 3.5, [3, 4.4, 5, 5.5], value_keys=['b', 'a'])},
            {'name': 'DictNoDef', 'raw_limit': RawLimit(2.25, 4.75, None, [3, 4.4, 5, 5.5], value_keys=['b', 'a'])},
            {'name': 'DictMin', 'raw_limit': RawLimit({'a': 2.25, 'b': 2.25}, 4.75, 3.5, [3, 4.4, 5, 5.5])},
            {'name': 'DictMax', 'raw_limit': RawLimit(2.25, {'a': 4.75, 'b': 4.75}, 3.5, [3, 4.4, 5, 5.5])},
            {'name': 'DictDef', 'raw_limit': RawLimit(2.25, 4.75, {'a': 3.5, 'b': 3.5}, [3, 4.4, 5, 5.5])},
            {'name': 'DictFix', 'raw_limit': RawLimit(2.25, 4.75, 3.5, [3, 4.4], {'b': [5, 5.5], 'a': [5, 5.5]})},
        ]

        for test in same_tests:
            self.add_test(test, TestSameLimitImport)

        scalar_tests = [
            {'name': 'ScalarKey', 'raw_limit': RawLimit(2.25, 4.75, 3.5, [3, 4.4, 5, 5.5], value_keys=[0])},
            {'name': 'ScalarCount', 'raw_limit': RawLimit(2.25, 4.75, 3.5, [3, 4.4, 5, 5.5], value_keys=1)},
            {'name': 'ScalarBase', 'raw_limit': RawLimit(2.25, 4.75, 3.5, [3, 4.4, 5, 5.5])},
            {'name': 'ScalarNoDef', 'raw_limit': RawLimit(2.25, 4.75, None, [3, 4.4, 5, 5.5])},
        ]

        for test in scalar_tests:
            self.add_test(test, TestScalarLimitImport)

        # TODO: Tests für Chooser

    def add_test(self, test, base_class):
        class CurrentTestLimitImport(base_class):
            pass

        CurrentTestLimitImport.init(**test)
        self.addTest(defaultTestLoader.loadTestsFromTestCase(CurrentTestLimitImport))
