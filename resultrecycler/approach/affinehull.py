#!/usr/bin/python

from numpy import array, dot, zeros, argmax, allclose
from numpy.core.umath import equal, absolute
from numpy.linalg import solve
from .basic import Approach, NearestNeighbor


class AffineHull(Approach):
    """
    In a 2D x-y-Diagram: Connects closest x values and draws a line between the y values. The y value at the
    required x is returned.

    Mathematically more correct: Find the closest coordinates, which produce an affine hull for the given coordinates
    and return the value above the given coordinates which lies in the plane of the affine hull.

    Read my thesis for more details...
    """
    def __init__(self):
        super().__init__()
        self.fallback_approach = NearestNeighbor()

    def guess(self, data, candidate):
        approach_base = AffineBase()
        sample_indices = []
        i = 0
        try:
            while approach_base.is_affine_independent(candidate.coordinates):
                next_nearest = data[i].coordinates
                if approach_base.is_affine_independent(next_nearest):
                    approach_base.add(next_nearest)
                    sample_indices.append(i)
                i += 1
        except IndexError:
            return self.fallback_approach.guess(data, candidate)
        coefficients = approach_base.affine_coefficients(candidate.coordinates)
        result = 0
        for i in range(len(coefficients)):
            values = data[sample_indices[i]].values
            result += (values * coefficients[i])
        return result


class AffineBase:
    def __init__(self):
        self.points = []
        self._current_candidate = None
        self._current_affine_coefficients = None
        self.independence_indices = []
        self.point_matrix = None
        self.base_matrix = None
        self._current_affine_combination = None

    def is_affine_independent(self, candidate):
        return not self.is_affine_dependent(candidate)

    def is_affine_dependent(self, candidate):
        self._evaluate(candidate)
        return self._current_candidate_dependent

    def affine_coefficients(self, candidate):
        self._evaluate(candidate)
        return self._current_affine_coefficients

    def _evaluate(self, candidate):
        if self._is_current_candidate(candidate):
            return
        self._current_candidate = candidate
        if self.dim() == 0:
            self._current_candidate_dependent = False
            self._current_affine_coefficients = None
        else:
            self._compute_and_verify_coefficients()

    def _is_current_candidate(self, candidate):
        return self._current_candidate is not None and all(equal(candidate, self._current_candidate))

    def _compute_and_verify_coefficients(self):
        prunedCandidate = self._pruned_with_affine_coefficient(self._current_candidate)
        self._current_affine_coefficients = solve(self.base_matrix, prunedCandidate)
        self._current_affine_combination = dot(self.point_matrix, self._current_affine_coefficients)
        self._current_candidate_dependent = allclose(self._current_affine_combination, self._current_candidate)

    def _pruned_with_affine_coefficient(self, vector):
        modified_vector = zeros(self.dim())
        modified_vector[0] = 1
        for i in range(1, self.dim()):
            modified_vector[i] = vector[self.independence_indices[i - 1]]
        return modified_vector

    def add(self, vector):
        self._evaluate(vector)
        if self._current_candidate_dependent:
            raise Exception
        if self.dim() > 0:
            self._add_independent_index()
        self._add_to_points(self._current_candidate)
        self._update_base()

    def _add_independent_index(self):
        affine_approximation = self._current_affine_combination
        independent_index = argmax(absolute(affine_approximation - self._current_candidate))
        self.independence_indices.append(independent_index)

    def _add_to_points(self, candidate):
        self.points.append(candidate)
        self.point_matrix = array(self.points).T

    def _update_base(self):
        new_base = zeros((self.dim(), self.dim()))
        i = 0
        for p in self.points:
            new_base[:, i] = self._pruned_with_affine_coefficient(p)
            i += 1
        self.base_matrix = new_base

    def __len__(self):
        return len(self.points)

    def dim(self):
        return len(self.points)
