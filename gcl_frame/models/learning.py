import numpy as np

from gcl_frame.utils.distances import euclid_d


def split_data(D, y, train_index, test_index):
	D_app = [D[i] for i in train_index]
	D_test = [D[i] for i in test_index]
	y_app = [y[i] for i in train_index]
	y_test = [y[i] for i in test_index]
	return D_app, D_test, y_app, y_test


def evaluate_D(D_app, y_app, D_test, y_test, mode='reg'):
	from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
	from gcl_frame.utils.distances import rmse, accuracy
	from sklearn.model_selection import GridSearchCV

	if mode == 'reg':
		knn = KNeighborsRegressor(metric='precomputed')
		scoring = 'neg_root_mean_squared_error'
		perf_eval = rmse
	else:
		knn = KNeighborsClassifier(metric='precomputed')
		scoring = 'accuracy'
		perf_eval = accuracy
	grid_params = {
		'n_neighbors': [3, 5, 7, 9, 11]
	}

	clf = GridSearchCV(
		knn, param_grid=grid_params,
		scoring=scoring,
		cv=5, return_train_score=True, refit=True
	)
	clf.fit(D_app, y_app)
	y_pred_app = clf.predict(D_app)
	y_pred_test = clf.predict(D_test)
	return perf_eval(y_pred_app, y_app), perf_eval(y_pred_test, y_test), clf


def xp_knn(
		Gn, y_all, y_distance=euclid_d,
		mode='reg', unlabeled=False, ed_method='BIPARTITE',
		descriptor='atom_bond_types',
		optim_method='jia2021', embedding_space='y',
		**kwargs
):
	"""
	Perform a knn regressor on given dataset
	"""
	import time
	from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

	from embed.embed import compute_D_embed

	start_time = time.time()

	stratified = False
	if mode == 'classif':
		stratified = True

	if stratified:
		rs = StratifiedShuffleSplit(n_splits=10, test_size=.1, random_state=0)
	else:
		# 		rs = ShuffleSplit(n_splits=10, test_size=.1) #, random_state=0)
		rs = ShuffleSplit(n_splits=10, test_size=.1, random_state=0)

	if stratified:
		split_scheme = rs.split(Gn, y_all)
	else:
		split_scheme = rs.split(Gn)

	results = []
	i = 1
	for train_index, test_index in split_scheme:
		print()
		print("Split {0}/{1}".format(i, 10))
		i = i + 1
		# Get splitted data
		G_app, G_test, y_app, y_test = split_data(
			Gn, y_all,
			train_index, test_index
		)

		if optim_method in ['embed', 'jia2021']:
			# Compute distances between elements in embedded space:
			res_embed = compute_D_embed(
				G_app, np.array(y_app), G_test, np.array(y_test),
				y_distance=y_distance,
				mode=mode, unlabeled=unlabeled, ed_method=ed_method,
				descriptor=descriptor,
				embedding_space=embedding_space,
				fit_test=(optim_method == 'embed'),
				n_classes=(len(np.unique(y_all)) if mode == 'classif' else None),
				**kwargs
			)
		else:
			res_embed = None

		if optim_method == 'embed':
			perf_app, perf_test, mat_app, mat_test, model = res_embed
			save_perf_for_embed(
				results, perf_app, perf_test,
				y_app, y_test, mat_app, mat_test, model
			)
		else:
			dis_mat_emded = res_embed
			evaluate_setup(
				G_app, y_app, G_test, y_test, dis_mat_emded, results,
				i, descriptor, optim_method, ed_method, unlabeled, mode,
				y_distance,
				**kwargs
			)

	# Print run time:
	run_time = time.time() - start_time
	print()
	print('Total running time of all CV trials: %.2f seconds.' % run_time)
	print(
		'Attention: the running time might be inaccurate when some parts of the '
		'code are executed a priori.'
	)

	return results


def evaluate_setup(
		G_app, y_app, G_test, y_test, dis_mat_emded, results, i,
		descriptor, optim_method, ed_method, unlabeled, mode, y_distance,
		**kwargs
):
	from ged import compute_D_random, compute_D_expert
	from ged import compute_D_fitted, compute_D_GH2020

	cur_results = {}
	cur_results['y_app'] = y_app
	cur_results['y_test'] = y_test
	cur_results['y_dis_emded'] = dis_mat_emded
	# Feed distances will all methods to compare
	edit_cost_fun = (
		'NON_SYMBOLIC' if (
				descriptor == '1hot-dis' or len(kwargs['node_attrs']) > 0 or len(kwargs['edge_attrs']) > 0
		) else 'CONSTANT'
	)
	distances = {}
	if optim_method == 'random':
		distances['random'] = compute_D_random(
			G_app, G_test, ed_method, edit_cost_fun=edit_cost_fun, **kwargs
		)
	elif optim_method == 'expert':
		distances['expert'] = compute_D_expert(
			G_app, G_test, ed_method, edit_cost_fun=edit_cost_fun, **kwargs
		)
	elif optim_method == 'jia2021':
		distances['fitted'] = compute_D_fitted(
			G_app, dis_mat_emded, G_test,
			y_distance=y_distance,
			mode=mode, unlabeled=unlabeled, ed_method=ed_method,
			edit_cost_fun=edit_cost_fun,
			**kwargs
		)
	kwargs['nb_trial'] = i - 1
	# @TODO: Change this as needed.
	# distances['Garcia-Hernandez2020'] = compute_D_GH2020(
	# 	G_app, G_test,
	# 	ed_method, **kwargs
	# )
	for setup in distances.keys():
		print("{0} Mode".format(setup))
		setup_results = {}
		D_app, D_test, edit_costs = distances[setup]
		setup_results['D_app'] = D_app
		setup_results['D_test'] = D_test
		setup_results['edit_costs'] = edit_costs
		print(edit_costs)
		perf_app, perf_test, clf = evaluate_D(
			D_app, y_app, D_test, y_test, mode
		)

		setup_results['perf_app'] = perf_app
		setup_results['perf_test'] = perf_test
		setup_results['clf'] = clf

		print(
			"Learning performance with {1} costs : {0:.2f}".format(
				perf_app, setup
			)
		)
		print(
			"Test performance with {1} costs : {0:.2f}".format(
				perf_test, setup
			)
		)
		cur_results[setup] = setup_results
	results.append(cur_results)


def save_perf_for_embed(
		results, perf_app, perf_test,
		y_app, y_test, mat_app, mat_test, model
):
	"""
	Save performance for embed method into results.
	"""
	cur_results = {}
	cur_results['y_app'] = y_app
	cur_results['y_test'] = y_test

	setup = 'embed'
	print("{0} Mode".format(setup))
	setup_results = {}
	setup_results['mat_app'] = mat_app
	setup_results['mat_test'] = mat_test
	setup_results['perf_app'] = perf_app
	setup_results['perf_test'] = perf_test
	setup_results['model'] = model
	# setup_results['model_pred'] = model_pred

	print(
		"Learning performance with {1} costs : {0:.2f}".format(
			perf_app, setup
		)
	)
	print(
		"Test performance with {1} costs : {0:.2f}".format(
			perf_test, setup
		)
	)
	cur_results[setup] = setup_results

	results.append(cur_results)
