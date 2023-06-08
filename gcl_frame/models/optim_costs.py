from ged import compute_geds
from gcl_frame.utils.distances import sum_squares, euclid_d
import numpy as np
# from tqdm import tqdm

import sys
import copy


# sys.path.insert(0, "../")


def optimize_costs_unlabeled(nb_cost_mat, dis_k_vec, MAX_SAMPLE=np.inf):
	"""
	Optimize edit costs to fit dis_k_vec according to edit operations in nb_cost_mat
	! take care that nb_cost_mat do not contains 0 lines
	:param nb_cost_mat: \in \mathbb{N}^{N x 6} encoding the number of edit operations for each pair of graph
	:param dis_k_vec: The N distances to fit
	"""
	import cvxpy as cp
	import numpy as np
	# MAX_SAMPLE = 1000
	nb_cost_mat_m = np.array([[x[0], x[1], x[3], x[4]] for x in nb_cost_mat])
	dis_k_vec = np.array(dis_k_vec)
	# dis_k_vec_norm = dis_k_vec/np.max(dis_k_vec)

	# import pickle
	# pickle.dump([nb_cost_mat, dis_k_vec], open('debug', 'wb'))
	N = nb_cost_mat_m.shape[0]
	sub_sample = np.random.permutation(np.arange(N))
	if MAX_SAMPLE < N:
		sub_sample = sub_sample[:MAX_SAMPLE]

	x = cp.Variable(nb_cost_mat_m.shape[1])
	# Use `cp.norm` instead of `cp.sum_square`.
	cost = cp.norm(
		(nb_cost_mat_m[sub_sample, :] @ x) - dis_k_vec[sub_sample]
	)
	prob = cp.Problem(cp.Minimize(cost), [x >= 0])
	try:
		print('Trying cp.ECOS solver...')
		# The tests on Redox dataset show that the solvers `cp.ECOS` and `cp.SCS`
		# yield similar results, where the default solver `cp.MOSEK` requires a
		# license (until 2023.03.21) (T_T). See `this answer
		# <https://stackoverflow.com/a/65526728/9360800>`_ for more details.
		prob.solve(solver=cp.ECOS)  # , verbose=True)  # @TODO: test only.
	except cp.error.SolverError as e:
		print('Solver did not work. Trying cp.SCS solver...')
		prob.solve(solver=cp.SCS, verbose=True)  # @TODO: test only.
	# prob.solve()
	print('status:', prob.status)

	edit_costs_new = [x.value[0], x.value[1], 0, x.value[2], x.value[3], 0]
	# edit_costs_new = [xi if xi > 0 else 0 for xi in edit_costs_new]
	residual = prob.value
	return edit_costs_new, residual


def optimize_costs_classif_unlabeled(nb_cost_mat, Y):
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


def optimize_costs_classif(nb_cost_mat, Y):
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


def optimize_costs(nb_cost_mat, dis_k_vec):
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
		x >= [0.01 for i in range(nb_cost_mat.shape[1])],
		np.array([1.0, 1.0, -1.0, 0.0, 0.0, 0.0]).T @ x >= 0.0,
		np.array([0.0, 0.0, 0.0, 1.0, 1.0, -1.0]).T @ x >= 0.0
	]
	prob = cp.Problem(cp.Minimize(cost), constraints)
	try:
		print('Trying cp.ECOS solver...')
		# The tests on Redox dataset show that the solvers `cp.ECOS` and `cp.SCS`
		# yield similar results, where the default solver `cp.MOSEK` requires a
		# license (until 2023.03.21) (T_T). See `this answer
		# <https://stackoverflow.com/a/65526728/9360800>`_ for more details.
		prob.solve(solver=cp.ECOS)  # , verbose=True)  # @TODO: test only.
	except cp.error.SolverError as e:
		print('Solver did not work. Trying cp.SCS solver...')
		prob.solve(solver=cp.SCS, verbose=True)  # @TODO: test only.
	print('status:', prob.status)

	edit_costs_new = x.value
	residual = prob.value

	# Notice that the returned residual is the distance instead of the squared
	# distance. You may want to revise the codes where this function is invoked.
	return edit_costs_new, residual


def compute_optimal_costs(
		G, dis_y, init_costs=[3, 3, 1, 3, 3, 1],
		y_distance=euclid_d,
		mode='reg', unlabeled=False,
		ed_method='BIPARTITE',
		edit_cost_fun='CONSTANT',
		ite_max=5,
		rescue_optim_failure=False,
		verbose=True,
		**kwargs
):
	N = len(dis_y)

	G_pairs = []
	distances_vec = []

	for i in range(N):
		for j in range(i + 1, N):
			G_pairs.append([i, j])
			distances_vec.append(dis_y[i, j])
	ged_vec_init, n_edit_operations = compute_geds(
		G_pairs, G, init_costs, ed_method, edit_cost_fun=edit_cost_fun,
		verbose=verbose, **kwargs
	)

	residual_list = [sum_squares(ged_vec_init, distances_vec)]
	print('initial residual:', residual_list[-1])

	if unlabeled:
		method_optim = optimize_costs_unlabeled
	else:
		method_optim = optimize_costs
	# if mode == 'reg':
	# 	if unlabeled:
	# 		method_optim = optimize_costs_unlabeled
	# 	else:
	# 		method_optim = optimize_costs
	#
	# elif mode == 'classif':
	# 	if unlabeled:
	# 		method_optim = optimize_costs_classif_unlabeled
	# 	else:
	# 		method_optim = optimize_costs_classif
	#
	# else:
	# 	raise ValueError('"mode" should be either "reg" or "classif".')

	edit_costs_old = copy.deepcopy(init_costs)
	# idx_sep = len(init_costs[0])
	for i in range(ite_max):
		if verbose:
			print('\nite', i + 1, '/', ite_max, ':')
		# compute GEDs and numbers of edit operations.
		edit_costs_new, residual = method_optim(
			np.array(n_edit_operations), distances_vec
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
				edit_costs_new = edit_costs_old.copy()
		else:
			edit_costs_old = edit_costs_new.copy()
		ged_vec, n_edit_operations = compute_geds(
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
