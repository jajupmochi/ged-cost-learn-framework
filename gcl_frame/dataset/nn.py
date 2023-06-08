"""
nn



@Author: linlin
@Date: 27.05.23
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.utils import from_networkx


class NetworkXGraphDataset(Dataset):
	def __init__(
			self, graph_list, target_list,
			node_label_names=None, edge_label_names=None,
			node_attr_names=None, edge_attr_names=None,
			ensure_node_feats=True,
	):
		"""
		Initialize a PyTorch Geometric compatible dataset from a list of NetworkX
		graphs.

		parameters
		----------
		graph_list : list
			List of NetworkX graphs.
		target_list : list
			List of targets.
		node_label_names : list
			List of symbolic node attribute names to be encoded as one-hot.
		edge_label_names : list
			List of symbolic edge attribute names to be encoded as one-hot.
		node_attr_names : list
			List of node attribute names to be included in the node feature matrix.
		edge_attr_names : list
			List of edge attribute names to be included in the edge feature matrix.
		"""
		super(NetworkXGraphDataset, self).__init__()

		# Get node and edge attribute names:
		node_label_names, edge_label_names = self.get_attribute_names(
			graph_list,
			node_attr_names=node_label_names,
			edge_attr_names=edge_label_names
		)

		# Get all unique node and edge attributes in the label names:
		unique_node_attrs, unique_edge_attrs = self.get_all_attributes(
			graph_list,
			node_attr_names=node_label_names,
			edge_attr_names=edge_label_names
		)

		self.data_list = []
		for graph, target in zip(graph_list, target_list):
			# Convert each NetworkX graph to a PyTorch Geometric Data object
			data = from_networkx(graph)

			# Process node labels:
			# Initialize x as an empty tensor:
			if len(node_label_names) != 0 or len(node_attr_names) != 0:
				# @TODO: This is a hack to temporarily fix the issue when x is
				# a attribute of the graph. It will be assigned directed to data.x.
				# This should be fixed in the future. The data.y will be assigned
				# to the data.y as well.
				if node_attr_names is not None and 'x' in node_attr_names:
					data.x = torch.unsqueeze(torch.tensor(
						[float(i) for i in data.x], dtype=torch.float), dim=1
					)
				else:
					data.x = torch.empty((data.num_nodes, 0), dtype=torch.float)
				# Concatenate the node feature matrix with the node labels:
				for attr_name in node_label_names:
					# Convert symbolic node labels to one-hot encodings:
					label_set = unique_node_attrs[attr_name]
					label_table = {label: i for i, label in enumerate(label_set)}
					one_hot = one_hot_encode_labels(data[attr_name], label_table)
					# Concatenate the one-hot encodings to the node feature matrix:
					data.x = torch.cat((data.x, one_hot), dim=-1)
				# Concatenate the node feature matrix with the node attributes:
				for attr_name in node_attr_names:
					if attr_name == 'x':
						continue
					data.x = torch.cat(
						(data.x, torch.unsqueeze(torch.tensor(
							[float(i) for i in data[attr_name]], dtype=torch.float), dim=1
						)),
						dim=-1
					)
			# If the node feature matrix is empty and ensure_node_feats is True,
			# then initialize it as a tensor of ones:
			elif ensure_node_feats:
				data.x = torch.ones((data.num_nodes, 1), dtype=torch.float)
			else:
				data.x = torch.empty((data.num_nodes, 0), dtype=torch.float)

			# Process edge labels:
			# Initialize edge_attr as an empty tensor:
			if len(edge_label_names) != 0 or len(edge_attr_names) != 0:
				data.edge_attr = torch.empty(
					(data.num_edges, 0), dtype=torch.float
				)
			# Concatenate the edge feature matrix with the edge labels:
			for attr_name in edge_label_names:
				# Convert symbolic edge labels to one-hot encodings:
				label_set = unique_edge_attrs[attr_name]
				label_table = {label: i for i, label in enumerate(label_set)}
				if attr_name in node_label_names:
					name_in_data = 'edge_' + attr_name
				else:
					name_in_data = attr_name
				one_hot = one_hot_encode_labels(data[name_in_data], label_table)
				# Concatenate the one-hot encodings to the edge feature matrix:
				data.edge_attr = torch.cat((data.edge_attr, one_hot), dim=-1)
			# Concatenate the edge feature matrix with the edge attributes:
			for attr_name in edge_attr_names:
				if attr_name in node_attr_names:
					name_in_data = 'edge_' + attr_name
				else:
					name_in_data = attr_name
				data.edge_attr = torch.cat(
					(data.edge_attr, data[name_in_data]), dim=-1
				)

			# Add the target to the data object:
			data.y = torch.tensor([target], dtype=torch.float)
			self.data_list.append(data)

		# Store the node and edge label names:
		self.node_label_names = node_label_names
		self.edge_label_names = edge_label_names
		# Store the node and edge attribute names:
		self.node_attr_names = node_attr_names
		self.edge_attr_names = edge_attr_names


	def len(self):
		return len(self.data_list)


	def get(self, idx):
		return self.data_list[idx]


	@staticmethod
	def get_attribute_names(
			graph_list, node_attr_names=None, edge_attr_names=None
	):
		if node_attr_names is None:
			node_attr_names = []
		elif node_attr_names == 'all':
			raise NotImplementedError('node_attr_names cannot be "all".')
			node_attr_names = list(
				set(
					[attr for graph in graph_list for node, attr in
					 graph.nodes(data=True)]
				)
			)
		if edge_attr_names is None:
			edge_attr_names = []
		elif edge_attr_names == 'all':
			raise NotImplementedError('edge_attr_names cannot be "all".')
			edge_attr_names = list(
				set(
					[attr for graph in graph_list for edge, attr in
					 graph.edges(data=True)]
				)
			)
		return node_attr_names, edge_attr_names


	def get_all_attributes(
			self, graph_list, node_attr_names=None, edge_attr_names=None
	):
		"""
		Return all unique attributes on nodes and edges in a list of graphs,
		given a list of attribute names. If an attribute name list is None,
		that dict is set to empty. If an attribute name list is 'all', return all
		attributes in that dict.

		Args:
			graph_list: a list of NetworkX graphs
			node_attr_names: a list of node attribute names
			edge_attr_names: a list of edge attribute names

		Returns:
			node_attr_dict: a dictionary of node attributes
			edge_attr_dict: a dictionary of edge attributes
		"""
		node_attr_dict = {}
		edge_attr_dict = {}

		node_attr_names, edge_attr_names = self.get_attribute_names(
			graph_list, node_attr_names, edge_attr_names
		)

		for attr_name in node_attr_names:
			node_attr_dict[attr_name] = set(
				[attr[attr_name] for graph in graph_list for node, attr in
				 graph.nodes(data=True)]
			)
		for attr_name in edge_attr_names:
			edge_attr_dict[attr_name] = set(
				[attr[attr_name] for graph in graph_list for n1, n2, attr in
				 graph.edges(data=True)]
			)

		return node_attr_dict, edge_attr_dict


def one_hot_encode_labels(labels, label_table):
    label_indices = [label_table[label] for label in labels]
    num_classes = len(label_table) + 1

    # Create an empty one-hot tensor
    one_hot_labels = torch.zeros(len(labels), num_classes)

    # Set the non-zero values based on the label indices
    one_hot_labels.scatter_(1, torch.tensor(label_indices).unsqueeze(1), 1)

    return one_hot_labels


if __name__ == '__main__':
	from gklearn.dataset import Dataset
	from gklearn.experiments import DATASET_ROOT

	ds = Dataset('MUTAG', root=DATASET_ROOT, verbose=True)
	Gn = ds.graphs
	y_all = ds.targets
	dataset = NetworkXGraphDataset(
		Gn, y_all,
		node_label_names=ds.node_labels, edge_label_names=ds.edge_labels,
		node_attr_names=ds.node_attributes, edge_attr_names=ds.edge_attributes
	)

	# Create a PyTorch Geometric data loader
	batch_size = 32
	shuffle = True  # Set to False if you want to preserve the order of graphs
	num_workers = 4  # Number of subprocesses for data loading
	loader = DataLoader(
		dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
	)

	for batch in loader:
		# Access the batched data and targets
		x = batch.x  # Node features
		edge_index = batch.edge_index  # Edge indices
		y = batch.y  # Prediction targets

	# Perform further operations on the batched data and targets
	# ...
