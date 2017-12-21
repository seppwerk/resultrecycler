#!/usr/bin/python

from resultrecycler.approach.affinehull import AffineHull
from resultrecycler.approach.derivative import FirstDerivative, SecondDerivative
from config import Information


class InsuffientDataException(AttributeError):
    def __init__(self, approach, config):
        super().__init__('Insuffient data: approach {} requires {} but only {} is available'.format(
            approach.name, approach.required_information, config.information))


class ApproachChooser:
    @classmethod
    def choose(cls, config, approach_class):
        if approach_class is not None and config.information < approach_class.required_information:
            raise InsuffientDataException(approach_class, config)
        elif approach_class is not None:
            return approach_class()
        else:
            if config.information >= Information.SecondDerivative:
                return SecondDerivative()
            elif config.information >= Information.FirstDerivative:
                return FirstDerivative()
            else:
                return AffineHull()
