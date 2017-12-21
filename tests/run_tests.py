#!/usr/bin/python

from unittest import TextTestRunner

from tests.test_approaches import ApproachTestSuite
from tests.test_jacobian_converter import JacobianConverterTestSuite
from tests.test_limit import LimitImportTestSuite
from tests.test_metrics import MetricTestSuite
from tests.test_resultrecycler import ResultRecyclerTestSuite
from tests.test_vector_converter import VectorConverterTestSuite

TextTestRunner().run(VectorConverterTestSuite())
TextTestRunner().run(JacobianConverterTestSuite())
TextTestRunner().run(MetricTestSuite())
TextTestRunner().run(ApproachTestSuite())
TextTestRunner().run(LimitImportTestSuite())
TextTestRunner().run(ResultRecyclerTestSuite())
