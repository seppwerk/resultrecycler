#!/usr/bin/python


class SampleData:
    """
    Container for sample data.
    """
    def __init__(self, coordinates, values, jacobian=None, hessian=None):
        """
        Constructor for sample data

        :param coordinates: a container for coordinates of length m
        :param values: a container for values of length n
        :param jacobian: a container for jacobians of length n x m (optional)
        :param hessian: a container for hessians of length n x m x m (optional)

        As a container most key based types are allowed: dict, list, ndarray. Jacobians and Hessians should be
        formatted such that it makes sense (but it is checked within the first iteration, anyway).
        """
        self.coordinates = coordinates
        self.values = values
        self.jacobian = jacobian
        self.hessian = hessian


class PureData:
    coordinate_converter = None
    value_converter = None

    @classmethod
    def prepare(cls, coordinate_converter, value_converter, **_):
        cls.coordinate_converter = coordinate_converter
        cls.value_converter = value_converter


class CandidateData(PureData):
    def __init__(self, coordinates):
        self.coordinates = self.coordinate_converter.read(coordinates)


class ValueData(PureData):
    def __init__(self, sample_data):
        self.coordinates = self.coordinate_converter.read(sample_data.coordinates)
        self.values = self.value_converter.read(sample_data.values)
        self.current_distance = 0

    def __lt__(self, other):
        return self.current_distance < other.current_distance


class JacobianData(ValueData):
    jacobian_converter = None

    @classmethod
    def prepare(cls, jacobian_converter, **kwargs):
        super().prepare(**kwargs)
        cls.jacobian_converter = jacobian_converter

    def __init__(self, sample_data):
        super().__init__(sample_data)
        self.jacobian = self.jacobian_converter.read_jac(sample_data.jacobian)


class HessianData(JacobianData):
    hessian_converter = None

    @classmethod
    def prepare(cls, hessian_converter, **kwargs):
        super().prepare(**kwargs)
        cls.hessian_converter = hessian_converter

    def __init__(self, sample_data):
        super().__init__(sample_data)
        self.hessian = self.hessian_converter.read_hess(sample_data.hessian)
