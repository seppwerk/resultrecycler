#!/usr/bin/python

from numpy import ones

from ..config import Information


class Approach:
    """
    Base class for approaches.
    """
    def __init__(self):
        """
        The required_information field at the first evaluation to ensure data fulfills the required properties.
        """
        self.required_information = Information.ValueOnly

    def guess(self, data, candidate):
        """
        This method is the entry for the actual guess approach.
        :param data: Contains an ordered list of available sample data. At least one sample is always present, when this
        method is called
        :param candidate: Contains the data of the item, for which the guess is currently evaluated.
        :return: The actual guess produced by this approach
        """
        raise NotImplementedError('Guess method of {} is not implemented'.format(self.__name__))


class Fixed(Approach):
    """
    Simple approach which just returns an array of zeros as a result.
    Only for demonstration and testing.
    """
    fixed_value = 0

    def guess(self, data, *_):
        return ones(len(data[0].values)) * self.fixed_value


class NearestNeighbor(Approach):
    """
    Returns the values of the closest given sample data.
    """
    name = 'Nearest Neighbor Approach'

    def guess(self, data, *_):
        return data[0].values
