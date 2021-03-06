#!/usr/bin/python

from numpy import array
from .typecheck import ConverterType, TypeCheck, WrongDimensionError, DifferentKeyError, UnsupportedTypeError


class DerivativeConverter:
    def __init__(self, dim_c, dim_v, keys_c, keys_v, coordinate_converter, value_converter, **_):
        self.dim_c = dim_c
        self.dim_v = dim_v
        self.keys_c = keys_c
        self.keys_v = keys_v
        self.jac_shape = (self.dim_v, self.dim_c)
        self.hess_shape = (self.dim_v, self.dim_c, self.dim_c)
        self.validate(coordinate_converter, value_converter)

    def read(self, value_reader, data):
        raise NotImplementedError('Read function for coordinate vector not implemented')

    def read_jac(self, jacobian):
        ret = array(self.read(self.read_jac_values, jacobian), ndmin=2)
        ret.shape = self.jac_shape
        return ret

    def read_hess(self, hessian):
        ret = array(self.read(self.read_hess_values, hessian), ndmin=3)
        ret.shape = self.hess_shape
        return ret

    def read_values(self, values):
        return values

    def read_jac_values(self, jacobian):
        return self.read_values(jacobian)

    def read_hess_values(self, hessian):
        return self.read_values(hessian)

    def validate(self, coordinate_converter, value_converter):
        if self.dim_c != coordinate_converter.dim:
            raise WrongDimensionError('Jacobian', 'coordinate', self.dim_c, coordinate_converter.dim)
        if self.dim_v != value_converter.dim:
            raise WrongDimensionError('Jacobian', 'value', self.dim_v, value_converter.dim)
        if self.keys_c != coordinate_converter.keys:
            raise DifferentKeyError('Jacobian', 'coordinate', self.keys_c, coordinate_converter.keys)
        if self.keys_v != value_converter.keys:
            raise DifferentKeyError('Jacobian', 'value', self.keys_v, value_converter.keys)

    @classmethod
    def select(cls, init_data, coordinate_converter, value_converter):
        data = {'init_data': init_data,
                'coordinate_converter': coordinate_converter,
                'value_converter': value_converter}

        if value_converter.dim == 1 and value_converter.type is ConverterType.scalar:
            if TypeCheck.is_scalar(init_data):
                return ScalarScalarConverter(**data)
            if coordinate_converter.type is ConverterType.enumerable:
                return EnumScalarConverter(**data)
            elif coordinate_converter.type is ConverterType.keys:
                return KeyScalarConverter(**data)
        elif value_converter.type is ConverterType.enumerable:
            if TypeCheck.is_scalar(init_data[0]):
                return ScalarEnumConverter(**data)
            elif coordinate_converter.type is ConverterType.enumerable:
                return EnumEnumConverter(**data)
            elif coordinate_converter.type is ConverterType.keys:
                return KeyEnumConverter(**data)
        elif value_converter.type is ConverterType.keys:
            if TypeCheck.is_scalar(init_data[next(iter(init_data))]):
                return ScalarKeyConverter(**data)
            elif coordinate_converter.type is ConverterType.enumerable:
                return EnumKeyConverter(**data)
            elif coordinate_converter.type is ConverterType.keys:
                return KeyKeyConverter(**data)
        raise UnsupportedTypeError('Jacobian Data could not be converted', init_data)


class ScalarCoordinatesConverter(DerivativeConverter):
    def __init__(self, **kwargs):
        super().__init__(dim_c=1, keys_c=(0, ), **kwargs)

class EnumCoordinatesConverter(DerivativeConverter):
    def __init__(self, dim_c, **kwargs):
        super().__init__(dim_c=dim_c, keys_c=tuple(range(dim_c)), **kwargs)

class KeyCoordinatesConverter(DerivativeConverter):
    def __init__(self, dict_c, **kwargs):
        super().__init__(dim_c=len(dict_c.keys()), keys_c=tuple(sorted(dict_c.keys())), **kwargs)

    def read_jac_values(self, jacobian_values):
        return [jacobian_values[key] for key in self.keys_c]

    def read_hess_values(self, hessian_values):
        return [hessian_values[key1, key2] for key1 in self.keys_c for key2 in self.keys_c]


class ScalarValuesConverter(DerivativeConverter):
    def __init__(self, **kwargs):
        super().__init__(dim_v=1, keys_v=(0, ), **kwargs)

    def read(self, value_reader, data):
        return [value_reader(data)]


class EnumValuesConverter(DerivativeConverter):
    def __init__(self, dim_v, **kwargs):
        super().__init__(dim_v=dim_v, keys_v=tuple(range(dim_v)), **kwargs)

    def read(self, coordinate_reader, data):
        return [coordinate_reader(line) for line in data]


class KeyValuesConverter(DerivativeConverter):
    def __init__(self, dict_v, **kwargs):
        super().__init__(dim_v=len(dict_v.keys()), keys_v=tuple(sorted(dict_v.keys())), **kwargs)

    def read(self, coordinate_reader, data):
        return [coordinate_reader(data[key]) for key in self.keys_v]


class ScalarScalarConverter(ScalarValuesConverter, ScalarCoordinatesConverter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class EnumScalarConverter(ScalarValuesConverter, EnumCoordinatesConverter):
    def __init__(self, init_data, **kwargs):
        super().__init__(dim_c=len(init_data), **kwargs)


class KeyScalarConverter(ScalarValuesConverter, KeyCoordinatesConverter):
    def __init__(self, init_data, **kwargs):
        super().__init__(dict_c=init_data, **kwargs)


class ScalarEnumConverter(EnumValuesConverter, ScalarCoordinatesConverter):
    def __init__(self, init_data, **kwargs):
        super().__init__(dim_v=len(init_data), **kwargs)


class EnumEnumConverter(EnumValuesConverter, EnumCoordinatesConverter):
    def __init__(self, init_data, **kwargs):
        super().__init__(dim_v=len(init_data), dim_c=len(init_data[0]), **kwargs)


class KeyEnumConverter(EnumValuesConverter, KeyCoordinatesConverter):
    def __init__(self, init_data, **kwargs):
        super().__init__(dim_v=len(init_data), dict_c=(init_data[0]), **kwargs)


class ScalarKeyConverter(KeyValuesConverter, ScalarCoordinatesConverter):
    def __init__(self, init_data, **kwargs):
        super().__init__(dict_v=init_data, **kwargs)


class EnumKeyConverter(KeyValuesConverter, EnumCoordinatesConverter):
    def __init__(self, init_data, **kwargs):
        super().__init__(dict_v=init_data, dim_c=len(init_data[next(iter(init_data))]), **kwargs)


class KeyKeyConverter(KeyValuesConverter, KeyCoordinatesConverter):
    def __init__(self, init_data, **kwargs):
        super().__init__(dict_v=init_data, dict_c=init_data[next(iter(init_data))], **kwargs)
