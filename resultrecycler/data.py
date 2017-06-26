#!/usr/bin/python


class RawData:
    def __init__(self, coordinates, values, jacobian=None, hessian=None):
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


class Data(PureData):
    coordinate_converter = None
    value_converter = None

    def __init__(self, raw_data):
        self.coordinates = self.coordinate_converter.read(raw_data.coordinates)
        self.values = self.value_converter.read(raw_data.values)
        self.current_distance = 0

    def __lt__(self, other):
        return self.current_distance < other.current_distance


class JacobianData(Data):
    jacobian_converter = None

    @classmethod
    def prepare(cls, jacobian_converter, **kwargs):
        super().prepare(**kwargs)
        cls.jacobian_converter = jacobian_converter

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.jacobian = self.jacobian_converter.read_jac(raw_data.jacobian)


class HessianData(JacobianData):
    hessian_converter = None

    @classmethod
    def prepare(cls, hessian_converter, **kwargs):
        super().prepare(**kwargs)
        cls.hessian_converter = hessian_converter

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.hessian = self.hessian_converter.read_hess(raw_data.hessian)
