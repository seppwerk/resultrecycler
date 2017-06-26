#!/usr/bin/python

import unittest

import numpy

from approach import basic
from resultrecycler import ResultRecycler


#TODO: sinnvollerer Name
class TestResultRecycler(unittest.TestCase):
    
    def setUp(self):
        self.old_eins = OldSample([2, 1], 1, 20, 30)
        self.old_zwei = OldSample([1, 2], 2, 66, 77)
        self.old_drei = OldSample([1000, 2000], 2.00005, 999, 888)
        self.rr = ResultRecycler(get_koordinates_old, get_values_old)
        self.rr.add_old_sample(self.old_eins)
        self.rr.add_old_sample(self.old_zwei)
        self.new_sample = NewSample([2, 2.00005, 2])
    
    def test_sorted_sample_history_indices_in_order_of_distance_to_new_sample(self):
        coords_of_new_sample = get_koordinates_new(self.new_sample)
        sample_history = [self.old_eins, self.old_zwei]
        
        approach = basic.Approach(coords_of_new_sample, sample_history, get_koordinates_old, get_values_old)
        result = numpy.argsort(approach._get_distances_by_applying_metric())
        expected_order = numpy.array([1, 0])
        self.assertTrue(numpy.array_equal(result, expected_order), "expected: " + str(expected_order) + " returned was: " + str(result))
    
        approach._sample_history.append(self.old_drei)
        approach._sample_history_length = 3
        result = numpy.argsort(approach._get_distances_by_applying_metric())
        expected_order = numpy.array([1, 0, 2])
        self.assertTrue(numpy.array_equal(result, expected_order), "expected: " + str(expected_order) + " returned was: " + str(result))
        
        approach._sample_history.append(OldSample([100, 2000], 2.00005, 999, 888))
        approach._sample_history_length = 4
        result = numpy.argsort(approach._get_distances_by_applying_metric())
        expected_order = numpy.array([1, 0, 3, 2])
        self.assertTrue(numpy.array_equal(result, expected_order), "expected: " + str(expected_order) + " returned was: " + str(result))
