from ged2 import compute_geds2
from gcl_frame.models.optimizers.ins_del import _optimize_costs_ins_del
from gcl_frame.utils.distances import sum_squares, euclid_d
import numpy as np
# from tqdm import tqdm

import copy


# sys.path.insert(0, "../")


def initialize_cost_maps(node_pairs, edge_pairs, init_costs=[3, 3, 1, 3, 3, 1]):
	"""
	Initialize the cost map for the graph edit distance. The cost maps are
	dictionaries with keys being tuples of labels and values being the costs.

	parameter
	---------
	node_pairs : list of tuples
		List of all possible pairs of node labels.
	edge_pairs : list of tuples
		List of all possible pairs of edge labels.
	init_costs : list
		List of initial costs. The order is [ins_node, del_node, sub_node,
		ins_edge, del_edge, sub_edge]. Default is [3, 3, 1, 3, 3, 1].

	return
	------
	node_costs : dict
		Dictionary of node costs.
	edge_costs : dict
		Dictionary of edge costs.
	"""
	node_costs = {}
	edge_costs = {}

	# initialize node costs:
	# If node is unlabeled, add " " (space) as the special label for unified costs:
	if len(node_pairs) == 0:
		node_costs[('', ' ')] = init_costs[0]  # insertion
		node_costs[(' ', '')] = init_costs[1]  # deletion
		node_costs[(' ', ' ')] = init_costs[2]  # substitution
	else:
		for pair in node_pairs:
			# If the first label in the pair is '', then it represents an insertion.
			if pair[0] == '':
				node_costs[pair] = init_costs[0]
			# If the second label in the pair is '', then it represents a deletion.
			elif pair[1] == '':
				node_costs[pair] = init_costs[1]
			# Otherwise, it represents a substitution.
			else:
				node_costs[pair] = init_costs[2]

	# initialize edge costs:
	# If edge is unlabeled, add " " (space) as the special label for unified costs:
	if len(edge_pairs) == 0:
		edge_costs[('', ' ')] = init_costs[3]  # insertion
		edge_costs[(' ', '')] = init_costs[4]  # deletion
		edge_costs[(' ', ' ')] = init_costs[5]  # substitution
	else:
		for pair in edge_pairs:
			# If the first label in the pair is '', then it represents an insertion.
			if pair[0] == '':
				edge_costs[pair] = init_costs[3]
			# If the second label in the pair is '', then it represents a deletion.
			elif pair[1] == '':
				edge_costs[pair] = init_costs[4]
			# Otherwise, it represents a substitution.
			else:
				edge_costs[pair] = init_costs[5]

	return node_costs, edge_costs


def optimize_costs_unlabeled2(nb_cost_mat, dis_k_vec):
	"""
	Optimize edit costs to fit dis_k_vec according to edit operations in nb_cost_mat
	! take care that nb_cost_mat do not contains 0 lines
	:param nb_cost_mat: \in \mathbb{N}^{N x 6} encoding the number of edit operations for each pair of graph
	:param dis_k_vec: The N distances to fit
	"""
	import cvxpy as cp
	import numpy as np
	MAX_SAMPLE = 1000
	nb_cost_mat_m = np.array([[x[0], x[1], x[3], x[4]] for x in nb_cost_mat])
	dis_k_vec = np.array(dis_k_vec)
	# dis_k_vec_norm = dis_k_vec/np.max(dis_k_vec)

	# import pickle
	# pickle.dump([nb_cost_mat, dis_k_vec], open('debug', 'wb'))
	N = nb_cost_mat_m.shape[0]
	sub_sample = np.random.permutation(np.arange(N))
	sub_sample = sub_sample[:MAX_SAMPLE]

	x = cp.Variable(nb_cost_mat_m.shape[1])
	cost = cp.sum_squares(
		(nb_cost_mat_m[sub_sample, :] @ x) - dis_k_vec[sub_sample]
	)
	prob = cp.Problem(cp.Minimize(cost), [x >= 0])
	prob.solve()
	edit_costs_new = [x.value[0], x.value[1], 0, x.value[2], x.value[3], 0]
	edit_costs_new = [xi if xi > 0 else 0 for xi in edit_costs_new]
	residual = prob.value
	return edit_costs_new, residual


def optimize_costs_classif_unlabeled2(nb_cost_mat, Y):
	"""
	Optimize edit costs to fit dis_k_vec according to edit operations in
	nb_cost_mat
	! take care that nb_cost_mat do not contains 0 lines
	:param nb_cost_mat: \in \mathbb{N}^{N x 6} encoding the number of edit
	operations for each pair of graph
	:param dis_k_vec: {-1,1}^N vector of common classes
	"""
	# import cvxpy as cp
	from ml import reg_log
	# import pickle
	# pickle.dump([nb_cost_mat, Y], open('debug', 'wb'))
	nb_cost_mat_m = np.array(
		[[x[0], x[1], x[3], x[4]]
		 for x in nb_cost_mat]
	)
	w, J, _ = reg_log(nb_cost_mat_m, Y, pos_contraint=True)
	edit_costs_new = [w[0], w[1], 0, w[2], w[3], 0]
	residual = J[-1]

	return edit_costs_new, residual


def optimize_costs_classif2(nb_cost_mat, Y):
	"""
		Optimize edit costs to fit dis_k_vec according to edit operations in nb_cost_mat
		! take care that nb_cost_mat do not contains 0 lines
		:param nb_cost_mat: \in \mathbb{N}^{N x 6} encoding the number of edit operations for each pair of graph
		:param dis_k_vec: {-1,1}^N vector of common classes
	"""
	# import pickle
	# pickle.dump([nb_cost_mat, Y], open("test.pickle", "wb"))
	from ml import reg_log
	w, J, _ = reg_log(nb_cost_mat, Y, pos_contraint=True)
	return w, J[-1]


# %% Main functions:


def _optimize_costs_solve(nb_cost_mat, dis_k_vec, tri_rule_list, verbose=False):
	"""
	Optimize edit costs to fit dis_k_vec according to edit operations in nb_cost_mat
	! take care that nb_cost_mat do not contains 0 lines
	:param nb_cost_mat: \in \mathbb{N}^{N x 6} encoding the number of edit operations for each pair of graph
	:param dis_k_vec: The N distances to fit

	Notes
	-----
	According to `CVXPY's official documentation
	<http://cvxr.com/cvx/doc/advanced.html#eliminating-quadratic-forms>`_:
	"One particular reformulation
	that we strongly encourage is to eliminate quadratic forms—that is, functions
	like sum_square, sum(square(.)) or quad_form—whenever it is possible to
	construct equivalent models using norm instead."

	When using the `NON-SYMBOLIC` cost method of GED and the `sum_square` function
	is used, it often happens that the
	problem is infeasible and the returned new edit costs are None. Due to this,
	the `norm` function is used instead. For more details, see this `answer
	<https://stackoverflow.com/a/65526728/9360800>`_.
	"""
	import cvxpy as cp

	x = cp.Variable(nb_cost_mat.shape[1])
	# Use `cp.norm` instead of `cp.sum_square`.
	cost = cp.norm((nb_cost_mat @ x) - dis_k_vec)
	constraints = [
		np.array(r_vec).T @ x >= 0.0 for r_vec in tri_rule_list
	]
	# constraints = [
	# 	x >= [0.01 for i in range(nb_cost_mat.shape[1])],  # @TODO
	# ] + constraints  # @TODO
	constraints = [
		x >= [0.01 for i in range(nb_cost_mat.shape[1])],  # @TODO
		# x <= [15 for i in range(nb_cost_mat.shape[1])],  # @TODO
	] + constraints  # @TODO

	try:
		print('Trying cp.ECOS solver...')
		prob = cp.Problem(cp.Minimize(cost), constraints)
		# The tests on Redox dataset show that the solvers `cp.ECOS` and `cp.SCS`
		# yield similar results, where the default solver `cp.MOSEK` requires a
		# license (until 2023.03.21) (T_T). See `this answer
		# <https://stackoverflow.com/a/65526728/9360800>`_ for more details.
		prob.solve(solver=cp.ECOS)  # , verbose=True)  # @TODO: test only.
	except cp.error.SolverError as e:
		print('Solver did not work. Trying cp.SCS solver...')
		prob = cp.Problem(cp.Minimize(cost), constraints)
		prob.solve(solver=cp.SCS, verbose=True)  # @TODO: test only.

	edit_costs_new = x.value
	residual = prob.value

	return edit_costs_new, residual


def optimize_costs2(
		nb_cost_mat, dis_k_vec, sorted_label_pairs,
		tria_rule_mode='each',  # @TODO # 'none', 'each', 'ensemble',
		optim_mode='substitution_2steps',  # @TODO # 'ins_del', 'all', 'ensemble'/'original', 'substitution_2steps'
		remove_zeros=False  # @TODO
):
	"""
	"""
	# Print the sum of number of columns in nb_cost_mat as the total number of
	# each edit operation over all pairs of graphs:
	print(
		'Total number of edit operations over all pairs of graphs: '
		'[ ' + '  '.join([str(i) for i in np.sum(nb_cost_mat, axis=0)]) + ' ]'
	)
	# Print the number of zeros, the total number, and their ratio in nb_cost_mat:
	print(
		'Number of zeros in nb_cost_mat: {} / {} = {:.2f}%'.format(
			np.sum(nb_cost_mat == 0), nb_cost_mat.size,
			np.sum(nb_cost_mat == 0) / nb_cost_mat.size * 100
		)
	)

	# Optimize all edit costs simultaneously:
	if optim_mode == 'all':
		return _optimize_costs_all(
			nb_cost_mat, dis_k_vec, sorted_label_pairs, tria_rule_mode, remove_zeros
		)
	elif optim_mode == 'ins_del':
		return _optimize_costs_ins_del(
			nb_cost_mat, dis_k_vec, sorted_label_pairs, tria_rule_mode, remove_zeros
		)
	elif optim_mode == 'ensemble' or optim_mode == 'original':
		from gcl_frame.models.optimizers.ensemble import optimize_costs_ensemble
		return optimize_costs_ensemble(
			nb_cost_mat, dis_k_vec, sorted_label_pairs, tria_rule_mode, remove_zeros
		)
	elif optim_mode == 'substitution_2steps':
		from gcl_frame.models.optimizers.substitution_2steps import \
			optimize_costs_substitution_2steps
		return optimize_costs_substitution_2steps(
			nb_cost_mat, dis_k_vec, sorted_label_pairs, tria_rule_mode, remove_zeros
		)
	else:
		raise ValueError("Unknown optim_mode: {}.".format(optim_mode))


def compute_optimal_costs2(
		G, y,
		init_costs=None,
		y_distance=euclid_d,
		mode='reg', unlabeled=False,
		ed_method='BIPARTITE',
		edit_cost_fun='CONSTANT',
		ite_max=5,  # @TODO: change it as needed.
		rescue_optim_failure=False,
		verbose=True,
		**kwargs
):
	if init_costs is None:
		# Initial costs.
		from gcl_frame.utils.graph import get_label_pairs
		nl_all = kwargs.get('nl_all')
		el_all = kwargs.get('el_all')
		node_pairs = get_label_pairs(nl_all, sort=True, extended=True)
		edge_pairs = get_label_pairs(el_all, sort=True, extended=True)
		init_costs = initialize_cost_maps(node_pairs, edge_pairs)

	N = len(y)

	G_pairs = []
	distances_vec = []

	for i in range(N):
		for j in range(i + 1, N):
			G_pairs.append([i, j])
			distances_vec.append(y_distance(y[i], y[j]))
	ged_vec_init, n_edit_operations, sorted_label_pairs, costs_vecs = compute_geds2(
		G_pairs, G, init_costs, ed_method, edit_cost_fun=edit_cost_fun,
		verbose=verbose, **kwargs
	)

	residual_list = [sum_squares(ged_vec_init, distances_vec)]
	print('initial residual:', residual_list[-1])

	if mode == 'reg':
		if unlabeled:
			method_optim = optimize_costs_unlabeled2
		else:
			method_optim = optimize_costs2

	elif mode == 'classif':
		if unlabeled:
			method_optim = optimize_costs_classif_unlabeled2
		else:
			method_optim = optimize_costs_classif2

	else:
		raise ValueError('"mode" should be either "reg" or "classif".')

	edit_costs_old = copy.deepcopy(init_costs)
	# idx_sep = len(init_costs[0])
	for i in range(ite_max):
		print('\nite', i + 1, '/', ite_max, ':')
		# compute GEDs and numbers of edit operations.
		edit_costs_new, residual = method_optim(
			np.array(n_edit_operations), distances_vec, sorted_label_pairs
		)
		# if the optimization fails:
		if edit_costs_new is None:
			if rescue_optim_failure:
				import warnings
				warnings.warn(
					'Oops (o_O)! The optimization of edit costs of this iteration '
					'(%d) seems failed and `edit_costs_new` returns `None`! '
					'Edit costs computed last iteration will be reused for the next '
					'iteration. This error may occur because the '
					'corrsponding CVXPY problem is infeasible. Please check '
					'`n_edit_operations` and `distance_vec` or ajust the constraints'
					'of the CVXPY porblem.' % (i + 1)
				)
				edit_costs_new = copy.deepcopy(edit_costs_old)
		else:
			edit_costs_old = copy.deepcopy(edit_costs_new)
		ged_vec, n_edit_operations, sorted_label_pairs, cost_vecs = compute_geds2(
			G_pairs, G, edit_costs_new, ed_method, edit_cost_fun=edit_cost_fun,
			verbose=verbose, **kwargs
		)
		print('new edit costs:', edit_costs_new)
		# idx_sep = len(edit_costs_new[0])
		residual_list.append(sum_squares(ged_vec, distances_vec))
		print('residual:', residual_list[-1])

	return edit_costs_new


def get_optimal_costs_GH2020(**kwargs):
	import pickle
	import os
	dir_root = 'cj/output/'
	ds_name = kwargs.get('ds_name')
	nb_trial = kwargs.get('nb_trial')
	file_name = os.path.join(
		dir_root, 'costs.' + ds_name + '.' + str(nb_trial) + '.pkl'
	)
	with open(file_name, 'rb') as f:
		edit_costs = pickle.load(f)
	return edit_costs



