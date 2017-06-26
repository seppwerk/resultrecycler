#!/usr/bin/python

from numpy import ones

from config import Information


class Approach:
    def __init__(self, config):
        self.config = config
        self.required_information = Information.ValueOnly

    def guess(self, *_):
        raise NotImplementedError('Guess method of {} is not implemented'.format(self.__name__))


class Fixed(Approach):
    fixed_value = 0

    def __init__(self, config):
        super().__init__(config)

    def guess(self, data, *_):
        return ones(self.config.value_converter.dim) * self.fixed_value


class NearestNeighbor(Approach):
    name = 'Nearest Neighbor Approach'

    def guess(self, data, *_):
        return data[0].values
