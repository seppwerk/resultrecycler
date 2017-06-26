#!/usr/bin/python

from numpy import zeros

from resultrecycler.approach import Approach


class WeightedSumApproach(Approach):
    def __init__(self, config):
        super().__init__(config)
        self.max_dimension = self.config.coordinate_converter.dim + 1

    def _get_normalized_weights(self, data):
        non_normalised_weights = self._get_non_normalized_weights(data)
        return non_normalised_weights / sum(non_normalised_weights)

    def _get_non_normalized_weights(self, data):
        number_of_relevant_samples = min(len(data), self.max_dimension)
        relevant_weights = zeros(number_of_relevant_samples)
        for key, dat in enumerate(data[:number_of_relevant_samples]):
            relevant_weights[key] = 1 / dat.current_distance
        return relevant_weights

    def guess(self, data, candidate):
        result = zeros(self.config.value_converter.dim)
        normalized_weights = self._get_normalized_weights(data)
        for key, weight in enumerate(normalized_weights):
            result += (data[key].values * weight)
        return result
