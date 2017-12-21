#!/usr/bin/python

from enum import Enum
from functools import total_ordering

from converter import VectorConverter, DerivativeConverter
from data import ValueData, JacobianData, HessianData, CandidateData


@total_ordering
class Information(Enum):
    ValueOnly = 0
    FirstDerivative = 1
    SecondDerivative = 2

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return


class Config:
    def __init__(self, init_data):
        self._init_information(init_data)
        self.candidate_class = CandidateData
        self.data_class = ValueData
        self.coordinate_converter = VectorConverter.select(init_data.coordinates, 'coordinates')
        self.value_converter = VectorConverter.select(init_data.values, 'values')
        self.jacobian_converter = None
        self.hessian_converter = None
        if self.information in (Information.FirstDerivative, Information.SecondDerivative):
            self.data_class = JacobianData
            self.jacobian_converter = DerivativeConverter.select(init_data.jacobian, self.coordinate_converter,
                                                                 self.value_converter)
            if self.information is Information.SecondDerivative:
                self.data_class = HessianData
                self.hessian_converter = self.jacobian_converter
        self.candidate_class.prepare(coordinate_converter=self.coordinate_converter, value_converter=self.value_converter)
        self.data_class.prepare(coordinate_converter=self.coordinate_converter, value_converter=self.value_converter,
                                jacobian_converter=self.jacobian_converter, hessian_converter=self.hessian_converter)

    def _init_information(self, init_data):
        if hasattr(init_data, 'hessian') and init_data.hessian is not None:
            self.information = Information.SecondDerivative
        elif hasattr(init_data, 'jacobian') and init_data.jacobian is not None:
            self.information = Information.FirstDerivative
        elif hasattr(init_data, 'values'):
            self.information = Information.ValueOnly
        else:
            raise AttributeError('Result has no attribute `values`')
