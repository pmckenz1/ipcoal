#!/usr/bin/env python

"""Sequentially Markov Coalescent (SMC) subpackage.

"""

import ipcoal.smc.likelihood

from ipcoal.smc.ms_smc import (
	get_genealogy_embedding_table,
	get_genealogy_embedding_edge_path,

	get_probability_tree_unchanged_given_b_and_tr,
	get_probability_tree_unchanged_given_b,

	get_probability_topology_unchanged_given_b_and_tr,
	get_probability_topology_unchanged_given_b,

	get_probability_no_change,
	get_probability_tree_change,
	get_probability_topology_change,
	get_probability_tree_unchanged,
	get_probability_topology_unchanged,	

	plot_edge_probabilities,
	plot_waiting_distance_distributions,

	get_expected_waiting_distance_to_recombination_event,
	get_expected_waiting_distance_to_no_change,
	get_expected_waiting_distance_to_tree_change,
	get_expected_waiting_distance_to_topology_change,

	get_waiting_distance_to_recombination_event_rv,
	get_waiting_distance_to_no_change_rv,	
	get_waiting_distance_to_tree_change_rv,
	get_waiting_distance_to_topology_change_rv,

	# methods to simulate many genealogies to get empirical and 
	# MS-SMC waiting distances for many embedded genealogies 
	# given a species tree (similar to those in the MS-SMC manuscript)

	# simulate_expected_waiting_distance_
	# 
)
