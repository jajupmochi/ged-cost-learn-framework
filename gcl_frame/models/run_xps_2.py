import sys


def run_xp(ds_name, output_file, mode, y_distance, ed_method, descriptor):
	from gklearn.dataset import Dataset
	from gklearn.experiments import DATASET_ROOT
	from learning2 import xp_knn2
	from gcl_frame.utils.graph import convert_sym_attrs_to_integers

	if ds_name.startswith('brem_togn'):
		from gcl_frame.dataset.get_data import get_get_data
		Gn, y_all = get_get_data(ds_name, descriptor)
		node_labels = list(Gn[0].nodes[list(Gn[0].nodes())[0]].keys())
		edge_labels = list(Gn[0].edges[list(Gn[0].edges())[0]].keys())
	else:
		ds = Dataset(ds_name, root=DATASET_ROOT, verbose=True)
		ds.remove_labels(
			node_attrs=ds.node_attrs, edge_attrs=ds.edge_attrs
		)  # @todo: ged can not deal with sym and unsym labels at the same time.
		# 	ds.cut_graphs(range(0, 20))  # This is for testing purposes.
		Gn = ds.graphs
		y_all = ds.targets
		node_labels = ds.node_labels
		edge_labels = ds.edge_labels

	# For the convenience of the experiments, we convert the labels to integers.
	Gn, nl_all, el_all = convert_sym_attrs_to_integers(
		Gn, return_attrs=True, to_str=True, remove_old_attrs=True, inplace=True
	)

	resu = {}
	resu['y_distance'] = y_distance
	resu['dataset'] = ds_name
	unlabeled = (len(node_labels) == 0 and len(edge_labels) == 0)
	results = xp_knn2(
		Gn, y_all, y_distance=y_distances[y_distance],
		mode=mode,
		unlabeled=unlabeled, ed_method=ed_method,
		node_labels=node_labels, edge_labels=edge_labels,
		nl_all=nl_all, el_all=el_all,
		ds_name=ds_name
	)
	resu['results'] = results
	resu['unlabeled'] = unlabeled
	resu['mode'] = mode
	resu['ed_method'] = ed_method
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
		Descriptor_List = ['atom_bond_types', '1hot', '1hot-dis']
		Edit_Cost_List = ['BIPARTITE', 'IPFP']
		Dataset_list = [
			'brem_togn_dGred', 'brem_togn_dGox', 'Alkane_unlabeled', 'Acyclic', 'Chiral', 'Vitamin_D',
			'Steroid', 'MUTAG'
		]
		Dis_List = ['euclidean', 'manhattan']
		task_grid = ParameterGrid(
			{
				'descriptor': Descriptor_List[0:1],
				'edit_cost': Edit_Cost_List[0:1],
				'dataset': Dataset_list[3:4],
				'distance': Dis_List[0:1],
			}
		)

		mode = 'reg'
		force_run = True  # @TODO: set as needed.
		# Run.
		for task in list(task_grid):
			print()
			print(task)

			output_result = 'outputs/results2.' + '.'.join(
				[
					task['dataset'], task['edit_cost'], task['distance'],
					task['descriptor']
				]
			) + '.pkl'
			# output_result = 'outputs/results2_GH2020.' + '.'.join(
			# 	[task['dataset'], task['edit_cost'], task['distance']]
			# ) + '.pkl'

			if not os.path.isfile(output_result) or force_run:
				resu, _ = run_xp(
					task['dataset'], output_result, mode,
					task['distance'], task['edit_cost'], task['descriptor']
				)
			else:
				resu = pickle.load(open(output_result, 'rb'))

			# Display results.
			from gcl_frame.dataset.compute_results import \
				organize_results_by_cost_settings, compute_displayable_results, \
				print_latex_results, print_average_edit_costs

			xps = ['fitted2']  # ["random", "expert", "fitted"] None
			results_by_xp = organize_results_by_cost_settings(resu, xps)
			p = compute_displayable_results(results_by_xp)
			print()

			import pprint
			pp = pprint.PrettyPrinter(indent=4)
			print('\n--------------------------------\n')
			pp.pprint(results_by_xp)
			print()
			pp.pprint(p)

			# Print average edit costs:
			print_average_edit_costs(resu['results'])

			# Print results in latex format:
			print_latex_results(p)

			print("Fini")

