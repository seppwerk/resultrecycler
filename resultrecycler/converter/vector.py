#!/usr/bin/python

from numpy import array
from converter.typecheck import TypeCheck, UnsupportedTypeError, ConverterType


class VectorConverter:
    @classmethod
    def select(cls, init_data, data_name):
        if TypeCheck.is_scalar(init_data):
            return ScalarConverter(init_data)
        elif TypeCheck.is_ndarray(init_data):
            return NdArrayConverter(init_data)
        elif TypeCheck.is_list(init_data):
            return ListConverter(init_data)
        elif TypeCheck.is_tuple(init_data):
            return TupleConverter(init_data)
        elif TypeCheck.is_dict(init_data):
            return DictConverter(init_data)
        else:
            raise UnsupportedTypeError(data_name, init_data)


class ScalarConverter:
    type = ConverterType.scalar

    def __init__(self, _):
        self.dim = 1
        self.keys = tuple(range(1))

    @classmethod
    def read(cls, value):
        return array(value, ndmin=1)

    @classmethod
    def export(cls, value):
        return value[0]


class EnumerableConverter:
    type = ConverterType.enumerable

    def __init__(self, init_data):
        self.dim = len(init_data)
        self.keys = tuple(range(self.dim))


class NdArrayConverter(EnumerableConverter):
    def __init__(self, init_data):
        super().__init__(init_data)

    @classmethod
    def read(cls, values):
        return array(values)

    @classmethod
    def export(cls, values):
        return array(values)


class ListConverter(EnumerableConverter):
    def __init__(self, init_data):
        super().__init__(init_data)

    @classmethod
    def read(cls, values):
        return array(values)

    @classmethod
    def export(cls, values):
        return list(values)


class TupleConverter(EnumerableConverter):
    def __init__(self, init_data):
        super().__init__(init_data)

    @classmethod
    def read(cls, values):
        return array(values)

    @classmethod
    def export(cls, values):
        return tuple(values)


class DictConverter:
    type = ConverterType.keys

    def __init__(self, init_data):
        self.keys = tuple(sorted(init_data.keys()))
        self.dim = len(self.keys)

    def read(self, values):
        return array([values[key] for key in self.keys])

    def export(self, values):
        return {key: values[index] for index, key in enumerate(self.keys)}
