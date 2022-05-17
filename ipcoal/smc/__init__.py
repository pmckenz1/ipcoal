#!/usr/bin/env python

"""Sequentially Markov Coalescent (SMC) subpackage.

"""


from .smc import (
	get_genealogy_embedding_table,
	get_genealogy_embedding_edge_path,
	get_probability_tree_unchanged_given_b_and_tr,
	get_probability_tree_unchanged_given_b,
	get_probability_topology_unchanged_given_b_and_tr,
	get_probability_topology_unchanged_given_b,
	get_probability_of_no_change,
	get_probability_of_tree_change,
	get_probability_of_topology_change,
	plot_edge_probabilities,
)
