#!/usr/bin/python

from approach import ApproachChooser, NearestNeighbor
from config import Config
from data import CandidateData
from limits import Limit, RawLimit
from metric import EuclidianMetric


class ResultRecycler:
    def __init__(self, metric=None, approach_class=None, limit=None):
        self._config = None
        self._data = []
        self._metric = EuclidianMetric() if metric is None else metric
        self._approach_class = approach_class
        self._limit = limit

    def calculate(self, new_coordinates):
        if not self._data:
            if self._limit is None:
                raise Exception('no information about result structure was given')
            else:
                return self._limit.default
        candidate_coordinates = CandidateData(new_coordinates)
        self._metric.calc_distances(candidate_coordinates, self._data)
        approach_values = self._approach_class.guess(self._data, candidate_coordinates)
        neighbor_values = self._nearest_neighbor_approach.guess(self._data)
        return self._limit.choose_values(approach_values, neighbor_values)

    def add_data(self, raw_data):
        if not self._data:
            self._configure(raw_data)
        self._data.append(self._config.data_class(raw_data))

    def _configure(self, raw_data):
        self._config = Config(raw_data)
        self._metric.init_weights(self._config)
        self._approach_class = ApproachChooser.choose(self._config, self._approach_class)
        self._nearest_neighbor_approach = NearestNeighbor(self._config)
        if self._limit is None:
            self._limit = Limit(RawLimit(value_keys=self._config.value_converter.keys))
