import sys


def run_xp(ds_name, output_file, mode, **tasks):
	from gklearn.dataset import Dataset
	from gklearn.experiments import DATASET_ROOT
	from learning import xp_knn

	if ds_name.startswith('brem_togn'):
		from gcl_frame.dataset.get_data import get_get_data
		Gn, y_all = get_get_data(ds_name, tasks['descriptor'])
		node_labels = list(Gn[0].nodes[list(Gn[0].nodes())[0]].keys())
		edge_labels = list(Gn[0].edges[list(Gn[0].edges())[0]].keys())
		node_attrs = []
		edge_attrs = []
	else:
		from gcl_frame.dataset.get_data import format_ds
		ds = Dataset(ds_name, root=DATASET_ROOT, verbose=True)
		ds = format_ds(ds, ds_name)
		# 	ds.cut_graphs(range(0, 20))  # This is for testing purposes.
		Gn = ds.graphs
		y_all = ds.targets
		node_labels = ds.node_labels
		edge_labels = ds.edge_labels
		node_attrs = ds.node_attrs
		edge_attrs = ds.edge_attrs

	resu = {}
	resu['task'] = task
	resu['dataset'] = ds_name
	unlabeled = (
			len(node_labels) == 0 and len(edge_labels) == 0
			and len(node_attrs) == 0 and len(edge_attrs) == 0
	)
	y_distance = (y_distances['classif'] if mode == 'classif' else y_distances[task['y_distance']])
	results = xp_knn(
		Gn, y_all, y_distance=y_distance,
		mode=mode,
		unlabeled=unlabeled, ed_method=task['edit_cost'],
		node_labels=node_labels, edge_labels=edge_labels,
		node_attrs=node_attrs, edge_attrs=edge_attrs,
		ds_name=ds_name,
		descriptor=task['descriptor'], optim_method=task['optim_method'],
		embedding_space=task['space'],
	)
	resu['results'] = results
	resu['unlabeled'] = unlabeled
	resu['mode'] = mode
	# resu['ed_method'] = ed_method
	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	pickle.dump(resu, open(output_file, 'wb'))

	return resu, output_result


def run_from_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"dataset", help="path to / name of the dataset to predict"
	)
	parser.add_argument(
		"output_file", help="path to file which will contains the results"
	)
	# parser.add_argument(
	# 	"-u", "--unlabeled",
	# 	help="Specify that the dataset is unlabeled graphs",
	# 	action="store_true"
	# )
	parser.add_argument(
		"-m", "--mode", type=str, choices=['reg', 'classif'],
		help="Specify if the dataset a classification or regression problem"
	)
	parser.add_argument(
		"-y", "--y_distance", type=str,
		choices=['euclidean', 'manhattan', 'classif'],
		default='euclid',
		help="Specify the distance on y  to fit the costs"
	)

	args = parser.parse_args()

	dataset = args.dataset
	output_result = args.output_file
	unlabeled = args.unlabeled
	mode = args.mode

	print(args)
	y_distances = {
		'euclidean': euclid_d,
		'manhattan': man_d,
		'classif': classif_d
	}
	y_distance = y_distances['euclid']

	run_xp(dataset, output_result, unlabeled, mode, y_distance)


if __name__ == "__main__":

	import pickle
	import os

	from gcl_frame.utils.distances import euclid_d, man_d, classif_d
	from gcl_frame.utils.utils import mode_from_dataset

	y_distances = {
		'euclidean': euclid_d,
		'manhattan': man_d,
		'classif': classif_d
	}

	# Read arguments.
	if len(sys.argv) > 1:
		run_from_args()
	else:
		from sklearn.model_selection import ParameterGrid

		# Get task grid.
		Dataset_list = [
			'Alkane_unlabeled', 'Acyclic',  # 0-1 Regression
			'QM7', 'QM9',  # 2-3 Regression: big
			'brem_togn_dGred', 'brem_togn_dGox',  # 4-5 Regression: Redox
			'MAO', 'PAH', 'MUTAG', 'Monoterpens',  # 6-9 Jia thesis: mols
			'Letter-high', 'Letter-med', 'Letter-low',  # 10-12 Jia thesis: letters
			'ENZYMES', 'AIDS', 'NCI1', 'NCI109', 'DD',  # 13-17 Jia thesis: bigger
			'Mutagenicity', 'IMDB-BINARY', 'COX2', 'PTC_MR',  # 18-21 Fuchs2022 PR paper
			# 'Chiral', 'Vitamin_D', 'Steroid'
		]
		Descriptor_List = ['atom_bond_types', 'mpnn', '1hot', '1hot-dis']
		Optim_Method_List = ['random', 'expert', 'jia2021', 'embed', 'GH2020']
		Space_List = [
			'y',
			'linear_reg', 'krr_rbf',
			'sp_kernel', 'structural_sp', 'path_kernel', 'treelet_kernel', 'wlsubtree_kernel',
			'mpnn', 'gcn', 'gat', 'gin', 'egnn', 'transformer', 'vae',
			'IPFP'
		]
		Edit_Cost_List = ['BIPARTITE', 'IPFP']
		Dis_List = ['euclidean', 'manhattan']

		task_grid = ParameterGrid(
			{
				'dataset': Dataset_list[10:11],
				'descriptor': Descriptor_List[0:1],
				'optim_method': Optim_Method_List[2:3],
				'space': Space_List[10:11],
				'edit_cost': Edit_Cost_List[0:1],
				'y_distance': Dis_List[0:1],
			}
		)

		# mode = 'reg'
		force_run = True
		# Run.
		for task in list(task_grid):
			print()
			print(task)

			mode = mode_from_dataset(task['dataset'])

			output_result = 'outputs/results.' + '.'.join(
				list(task.values())
			) + '.pkl'

			if not os.path.isfile(output_result) or force_run:
				resu, _ = run_xp(
					task['dataset'], output_result, mode,
					**{k: v for k, v in task.items() if k != 'dataset'}
				)
			else:
				resu = pickle.load(open(output_result, 'rb'))

			# Display results.
			from gcl_frame.dataset.compute_results import \
				organize_results_by_cost_settings, compute_displayable_results, \
				print_latex_results, print_average_edit_costs

			xps = ['random', 'expert', 'fitted', 'embed']
			results_by_xp = organize_results_by_cost_settings(resu, xps)
			p = compute_displayable_results(results_by_xp)

			import pprint
			pp = pprint.PrettyPrinter(indent=4)
			print('\n--------------------------------\n')
			pp.pprint(results_by_xp)
			print()
			pp.pprint(p)

			# Print average edit costs:
			if task['optim_method'] != 'embed':
				print_average_edit_costs(resu['results'])

			# Print results in latex format:
			print_latex_results(p, mode, rm_valid=True)

			print("Fini")

