#!/usr/bin/python

from unittest import TextTestRunner

from tests.test_approach.test_approaches import ApproachTestSuite
from tests.test_converter.test_jacobian import JacobianConverterTestSuite
from tests.test_converter.test_vector import VectorConverterTestSuite
from tests.test_limit import LimitImportTestSuite
from tests.test_metric.test_metrics import MetricTestSuite
from tests.test_resultrecycler import ResultRecyclerTestSuite

TextTestRunner().run(VectorConverterTestSuite())
TextTestRunner().run(JacobianConverterTestSuite())
TextTestRunner().run(MetricTestSuite())
TextTestRunner().run(ApproachTestSuite())
TextTestRunner().run(LimitImportTestSuite())
TextTestRunner().run(ResultRecyclerTestSuite())