from gcl_frame.utils.distances import euclid_d
from gklearn.ged.util import pairwise_ged2, get_nb_edit_operations2
from gklearn.utils import get_iters

import sys


def compute_ged2(
		Gi, Gj, edit_cost, edit_cost_fun='CONSTANT', method='BIPARTITE',
		repeats=1,  # @TODO: change as needed.
		return_sorted_costs=True, **kwargs
):
	"""
	Compute GED between two graph according to edit_cost.
	"""
	ged_options = {
		'edit_cost': edit_cost_fun,
		'method': method,
		'edit_cost_constants': edit_cost
	}
	node_labels = kwargs.get('node_labels', [])
	edge_labels = kwargs.get('edge_labels', [])
	dis, pi_forward, pi_backward = pairwise_ged2(
		Gi, Gj, ged_options, repeats=repeats
	)
	n_eo_tmp, sorted_label_pairs, sorted_costs = get_nb_edit_operations2(
		Gi, Gj, pi_forward, pi_backward, edit_cost=edit_cost_fun,
		node_labels=node_labels, edge_labels=edge_labels,
		edit_cost_constants=edit_cost,
		return_sorted_costs=return_sorted_costs
	)

	#  @TODO: for test only:
	# Assert if the multiplication of the number of edit operations and the
	# sorted costs is equal to the computed distance:
	try:
		import numpy as np
		dis_computed = np.matmul(n_eo_tmp, sorted_costs)
		assert dis == dis_computed
	except AssertionError:
		# print('AssertionError: dis != dis_computed')
		# print('dis:', dis)
		# print('dis_computed:', dis_computed)
		# print('n_eo_tmp: ', n_eo_tmp)
		# print('sorted_costs: ', sorted_costs)
		# print('Trying `np.isclose()` instead...')
		assert np.isclose(dis, dis_computed)

	return dis, n_eo_tmp, sorted_label_pairs, sorted_costs


def compute_ged_all_dataset(Gn, edit_cost, ed_method, **kwargs):
	N = len(Gn)
	G_pairs = []
	for i in range(N):
		for j in range(i, N):
			G_pairs.append([i, j])
	return compute_geds2(G_pairs, Gn, edit_cost, ed_method, **kwargs)


def compute_geds2(
		G_pairs, Gn, edit_cost, ed_method, edit_cost_fun='CONSTANT',
		verbose=True, **kwargs
):
	"""
	Compute GED between all indexes in G_pairs given edit_cost
	:return: ged_vec : the list of computed distances, n_edit_operations : the list of edit operations
	"""
	ged_vec = []
	n_edit_operations = []
	costs_vecs = []  # The sorted costs of the edit operations.
	for k in get_iters(
			range(len(G_pairs)), desc='Computing GED', file=sys.stdout,
			length=len(G_pairs), verbose=verbose
	):
		[i, j] = G_pairs[k]
		dis, n_eo_tmp, sorted_label_pairs, sorted_costs = compute_ged2(
			Gn[i], Gn[j], edit_cost=edit_cost, edit_cost_fun=edit_cost_fun,
			method=ed_method, **kwargs
		)
		ged_vec.append(dis)
		n_edit_operations.append(n_eo_tmp)
		costs_vecs.append(sorted_costs)

	return ged_vec, n_edit_operations, sorted_label_pairs, costs_vecs


def compute_D2(
		G_app, edit_cost, G_test=None, ed_method='BIPARTITE',
		edit_cost_fun='CONSTANT',
		**kwargs
):
	import numpy as np
	N = len(G_app)
	D_app = np.zeros((N, N))

	for i, G1 in get_iters(
			enumerate(G_app), desc='Computing D - app', file=sys.stdout,
			length=N
	):
		for j, G2 in enumerate(G_app[i + 1:], i + 1):
			D_app[i, j], _, _, _ = compute_ged2(
				G1, G2, edit_cost, method=ed_method, edit_cost_fun=edit_cost_fun,
				**kwargs
			)
			D_app[j, i] = D_app[i, j]
	if (G_test is None):
		return D_app, edit_cost
	else:
		D_test = np.zeros((len(G_test), N))
		for i, G1 in get_iters(
				enumerate(G_test), desc='Computing D - test', file=sys.stdout,
				length=len(G_test)
		):
			for j, G2 in enumerate(G_app):
				D_test[i, j], _, _, _ = compute_ged2(
					G1, G2, edit_cost, method=ed_method, edit_cost_fun=edit_cost_fun,
					**kwargs
				)
		return D_app, D_test, edit_cost


def compute_D_random2(
		G_app, G_test=None, ed_method='BIPARTITE',
		edit_cost_fun='CONSTANT', **kwargs
):
	import numpy as np
	edit_costs = np.random.rand(6)
	return compute_D2(
		G_app, edit_costs, G_test, ed_method=ed_method,
		edit_cost_fun=edit_cost_fun,
		**kwargs
	)


def compute_D_expert2(
		G_app, G_test=None, ed_method='BIPARTITE', edit_cost_fun='CONSTANT',
		**kwargs
):
	edit_cost = [3, 3, 1, 3, 3, 1]
	return compute_D2(
		G_app, edit_cost, G_test, ed_method=ed_method,
		edit_cost_fun=edit_cost_fun,
		**kwargs
	)


def compute_D_fitted2(
		G_app, y_app, G_test=None, y_distance=euclid_d,
		mode='reg', unlabeled=False, ed_method='BIPARTITE',
		edit_cost_fun='CONSTANT',
		**kwargs
):
	from optim_costs2 import compute_optimal_costs2

	costs_optim = compute_optimal_costs2(
		G_app, y_app, y_distance=y_distance,
		mode=mode, unlabeled=unlabeled, ed_method=ed_method,
		edit_cost_fun=edit_cost_fun,
		**kwargs
	)
	return compute_D2(
		G_app, costs_optim, G_test, ed_method=ed_method,
		edit_cost_fun=edit_cost_fun,
		**kwargs
	)


def compute_D_GH2020_2(
		G_app, G_test=None, ed_method='BIPARTITE', edit_cost_fun='CONSTANT',
		**kwargs
):
	from optim_costs import get_optimal_costs_GH2020
	costs_optim = get_optimal_costs_GH2020(**kwargs)
	return compute_D2(
		G_app, costs_optim, G_test, ed_method=ed_method,
		edit_cost_fun=edit_cost_fun,
		**kwargs
	)
