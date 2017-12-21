#!/usr/bin/python

from enum import Enum
from numpy import ndarray


class ConverterType(Enum):
    scalar = 0
    enumerable = 1
    keys = 2


class UnsupportedTypeError(NotImplementedError):
    def __init__(self, label, data, *args, **kwargs):
        message = 'Unsupported type for {}: {} with content {}'.format(label, type(data), data)
        super().__init__(message, *args, **kwargs)


class WrongDimensionError(AttributeError):
    def __init__(self, section, field, res, exp, *args, **kwargs):
        message = '{}: Wrong {} dimension: {} instead of {}'.format(section, field, res, exp)
        super().__init__(message, *args, **kwargs)


class DifferentKeyError(AttributeError):
    def __init__(self, section, field, res, exp, *args, **kwargs):
        message = '{}: Wrong {} keys: {} instead of {}'.format(section, field, res, exp)
        super().__init__(message, *args, **kwargs)


class TypeCheck:
    @classmethod
    def is_int(cls, obj):
        return isinstance(obj, int)

    @classmethod
    def is_scalar(cls, obj):
        return isinstance(obj, (int, float))

    @classmethod
    def is_ndarray(cls, obj):
        return isinstance(obj, ndarray)

    @classmethod
    def is_list(cls, obj):
        return isinstance(obj, list)

    @classmethod
    def is_tuple(cls, obj):
        return isinstance(obj, tuple)

    @classmethod
    def is_list_or_tuple(cls, obj):
        return isinstance(obj, (list, tuple))

    @classmethod
    def is_enumerable(cls, obj):
        return isinstance(obj, (list, tuple, ndarray))

    @classmethod
    def is_dict(cls, obj):
        return isinstance(obj, dict)