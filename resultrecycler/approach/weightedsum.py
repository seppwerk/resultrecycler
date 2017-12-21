#!/usr/bin/python

from numpy import zeros

from resultrecycler.approach import Approach


class WeightedSumApproach(Approach):
    """
    Weights the n+1 closest samples according to their distance to the given data.
    n is the dimension of the coordinates.
    This approach has not been evaluated properly, so far.
    """
    def __init__(self):
        super().__init__()
        self.value_dimension = None
        self.max_samples = None

    def _get_normalized_weights(self, data):
        non_normalised_weights = self._get_non_normalized_weights(data)
        return non_normalised_weights / sum(non_normalised_weights)

    def _get_non_normalized_weights(self, data):
        number_of_relevant_samples = min(len(data), self.max_samples)
        relevant_weights = zeros(number_of_relevant_samples)
        for key, dat in enumerate(data[:number_of_relevant_samples]):
            relevant_weights[key] = 1 / dat.current_distance
        return relevant_weights

    def guess(self, data, candidate):
        if self.value_dimension is None:
            self.value_dimension = len(data[0].values)
            self.max_samples = len(data[0].coordinates) + 1
        result = zeros(self.value_dimension)
        normalized_weights = self._get_normalized_weights(data)
        for key, weight in enumerate(normalized_weights):
            result += (data[key].values * weight)
        return result
