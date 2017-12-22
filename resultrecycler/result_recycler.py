#!/usr/bin/python

from .approach import ApproachChooser, NearestNeighbor
from .config import Config
from .data import CandidateData
from .limits import Limit, RawLimit
from .metrics import EuclidianMetric


class ResultRecycler:
    def __init__(self, metric=None, approach_class=None, limit=None):
        """
        :param metric: Object of metric to be used. EuclidianMetric will be used, if nothing else given.
        :param approach_class: Class of approach to be used. Automatically choosen according to given data, if unset.
        :param limit: Limits to be taken into account.
        """
        self._config = None
        self._data = []
        self._metric = EuclidianMetric() if metric is None else metric
        self._approach_class = approach_class
        self._limit = limit

    def calculate(self, new_coordinates):
        """
        Returns a guess, which could be matching values for the given coordinates according to the given data.
        :param new_coordinates: The coordinates to be evaluated, should have the same format as the coordinates in
         sample data.
        :return: A guess for the values in the same format as the sample data.
        """
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

    def add_data(self, sample_data):
        """
        Adds a new set of sample data. Should always have the same format (dimensions, derivatives, etc.)
        Will complete the configuration of result recycler on first run.
        :param sample_data: The new set of sample data
        """
        if not self._data:
            self._configure(sample_data)
        self._data.append(self._config.data_class(sample_data))

    def _configure(self, sample_data):
        self._config = Config(sample_data)
        self._metric.init_weights(self._config)
        self._approach_class = ApproachChooser.choose(self._config, self._approach_class)
        self._nearest_neighbor_approach = NearestNeighbor()
        if self._limit is None:
            self._limit = Limit(RawLimit(value_keys=self._config.value_converter.keys))
