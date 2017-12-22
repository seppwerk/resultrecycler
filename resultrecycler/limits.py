#!/usr/bin/python

import sys

from numpy import array, ones

from .converter.typecheck import TypeCheck


class DefaultValues:
    MINIMUM = -sys.maxsize
    MAXIMUM = sys.maxsize
    DEFAULT = 0


class RawLimit:
    def __init__(self, minimum=None, maximum=None, default=None,
                 common_fixed_values=None, individual_fixed_values=None, value_keys=None):
        self.minimum = minimum
        self.maximum = maximum
        self.default = default
        self.common_fixed_values = [] if common_fixed_values is None else common_fixed_values
        self.individual_fixed_values = individual_fixed_values
        self.value_keys = value_keys


class Limit:
    def __init__(self, raw_limit):
        self.value_keys = self._init_value_keys(raw_limit)
        self._init_bounds(raw_limit)
        self.fixed_values = self._init_fixed_values(raw_limit)
        self._adjust_bounds()

    def _init_value_keys(self, raw_limit):
        if raw_limit.value_keys is not None:
            return self._get_given_value_keys(raw_limit.value_keys)
        return self._get_value_keys_from_fields(raw_limit)

    @classmethod
    def _get_given_value_keys(cls, value_keys):
        if TypeCheck.is_enumerable(value_keys):
            return sorted(value_keys)
        elif TypeCheck.is_int(value_keys):
            return list(range(value_keys))
        raise Exception('Invalid value keys for limit: {}'.format(value_keys))

    @classmethod
    def _get_value_keys_from_fields(cls, raw_limit):
        fields_to_check = [raw_limit.minimum, raw_limit.maximum, raw_limit.default, raw_limit.individual_fixed_values]
        for field in fields_to_check:
            if field is not None and not TypeCheck.is_scalar(field):
                return cls._get_value_keys_from_field(field)
        for field in fields_to_check:
            if TypeCheck.is_scalar(field):
                return [0]
        raise Exception('Value keys missing')

    @classmethod
    def _get_value_keys_from_field(cls, iterable):
        if TypeCheck.is_enumerable(iterable):
            return list(range(len(iterable)))
        if TypeCheck.is_dict(iterable):
            return sorted(iterable.keys())

    def _init_bounds(self, raw_limit):
        self.minimum = self._evaluate_field(raw_limit.minimum)
        self.maximum = self._evaluate_field(raw_limit.maximum)
        self.default = self._evaluate_field(raw_limit.default)

    def _evaluate_field(self, field):
        if field is None:
            return None
        elif TypeCheck.is_scalar(field):
            return ones(len(self.value_keys)) * field
        else:
            return array([field[key] for key in self.value_keys])

    def _init_fixed_values(self, raw_limit):
        ret = [[] for _ in self.value_keys]
        for index, key in enumerate(self.value_keys):
            if self.minimum is not None and self.minimum[index] > DefaultValues.MINIMUM:
                ret[index].append(self.minimum[index])
            if self.maximum is not None and self.maximum[index] < DefaultValues.MAXIMUM:
                ret[index].append(self.maximum[index])
            ret[index].extend(raw_limit.common_fixed_values)
            if raw_limit.individual_fixed_values is not None:
                ret[index].extend(raw_limit.individual_fixed_values[key])
        return ret

    def _adjust_bounds(self):
        if self.default is None and self.minimum is not None and self.maximum is not None:
            self.default = self.minimum + (self.maximum - self.minimum) / 2.0
        elif self.default is None:
            self.default = ones(len(self.value_keys)) * DefaultValues.DEFAULT
        if self.minimum is None:
            self.minimum = ones(len(self.value_keys)) * DefaultValues.MINIMUM
        if self.maximum is None:
            self.maximum = ones(len(self.value_keys)) * DefaultValues.MAXIMUM

    def choose_values(self, approach_values, neighbor_values):
        if approach_values is None:
            return neighbor_values
        ret = approach_values
        for key in self.value_keys:
            if neighbor_values is not None and neighbor_values[key] in self.fixed_values[key]:
                ret[key] = neighbor_values[key]
            if approach_values[key] < self.minimum[key]:
                ret[key] = self.minimum[key]
            if approach_values[key] > self.maximum[key]:
                ret[key] = self.maximum[key]
        return ret
