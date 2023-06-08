"""
model_selection



@Author: linlin
@Date: 20.05.23
"""

import numpy as np
import matplotlib

# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, \
	ParameterGrid

import multiprocessing
from multiprocessing import Pool, Array
from functools import partial
import sys
import os
import time
import datetime
from gklearn.utils.graphfiles import loadDataset
from gklearn.utils.iters import get_iters

import torch
from torch_geometric.loader import DataLoader


def predict(
		data_loader, model, metric, device, model_type='reg', y_scaler=None,
		return_embeddings=False, n_classes=None,
		**kwargs
):
	model.eval()
	y_pred, y_true, embeddings = [], [], []
	with torch.no_grad():
		for data in data_loader:
			data = data.to(device)
			target = data.y
			output = model(data)
			if return_embeddings:
				cur_y = output[0].cpu().numpy()  # @TODO: detach ?
				embeddings.append(output[1].cpu().numpy())
			else:
				cur_y = output.cpu().numpy()
			if model_type == 'classif':
				if n_classes == 2: # for sigmoid output
					cur_y = cur_y.round()
				else:
					cur_y = np.argmax(cur_y, axis=1)  # for (log_)softmax output
					# @TODO: other possible activations?
			y_pred.append(cur_y)
			y_true.append(target.cpu().numpy())
	y_pred = np.reshape(np.concatenate(y_pred, axis=0), (-1, 1))
	y_true = np.reshape(np.concatenate(y_true, axis=0), (-1, 1))
	if return_embeddings:
		embeddings = np.concatenate(embeddings, axis=0)
	if y_scaler is not None:
		if model_type == 'classif':
			# Convert to int before inverse transform:
			y_pred = np.ravel(y_pred.astype(int))
			y_true = np.ravel(y_true.astype(int))
		y_pred = y_scaler.inverse_transform(y_pred)
		y_true = y_scaler.inverse_transform(y_true)
	if metric == 'rmse':
		from sklearn.metrics import mean_squared_error
		score = mean_squared_error(y_true, y_pred)
		score = np.sqrt(score)
	elif metric == 'mae':
		from sklearn.metrics import mean_absolute_error
		score = mean_absolute_error(y_true, y_pred)
	elif metric == 'accuracy':
		from sklearn.metrics import accuracy_score
		score = accuracy_score(y_true, y_pred)
	else:
		raise ValueError('"metric" must be either "rmse", "mae" or "accuracy".')
	if return_embeddings:
		return score, y_pred, y_true, embeddings
	else:
		return score, y_pred, y_true


def test_epoch(test_loader, model, criterion):
	model.eval()
	test_loss = 0
	with torch.no_grad():
		for data, target in test_loader:
			data = data.to(device)
			target = target.to(device)
			output = model(data)
			test_loss += criterion(output, target).item() * data.size(0)
	test_loss /= len(test_loader.dataset)
	return test_loss


def train_epoch(
		train_loader, model, optimizer, criterion, device, return_embeddings
):
	model.train()
	train_loss = 0
	for data in train_loader:
		data = data.to(device)
		x = data.x
		target = data.y
		optimizer.zero_grad()
		output = model(data)
		if return_embeddings:
			output = output[0]
		if criterion.__class__.__name__ == 'NLLLoss':
			# Convert target to int for NLLLoss:
			target = torch.flatten(target.long())
		elif criterion.__class__.__name__ == 'MSELoss':
			target = torch.unsqueeze(target, dim=1)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		train_loss += loss.item() * target.size(0)
	train_loss /= len(train_loader.dataset)
	return train_loss


def fit_model(
		train_loader,
		estimator,
		params,
		model_type,
		n_epochs,
		device,
		valid_loader=None,
		return_embeddings=False,
		lr=0.01, weight_decay=5e-4, patience=10,
		plot_loss=False,
		print_interval=10,  # @TODO: to change as needed.
		verbose=False,
		**kwargs
):
	# initialize the model:
	model = estimator(
		in_feats=train_loader.dataset.num_node_features,
		edge_dim=train_loader.dataset.num_edge_features,
		normalize=True,
		bias=True,
		feat_drop=0.,
		kernel_drop=0.,
		predictor_batchnorm=False,  # @TODO #(True if model_type == 'classif' else False),
		n_classes=kwargs.get('n_classes'),
		return_embeddings=return_embeddings,
		mode=('regression' if model_type == 'reg' else 'classification'),
		device=device,
		**params
	).to(device)
	print(model)

	# Choose the loss function:
	if model_type == 'reg':
		loss = torch.nn.MSELoss()
	elif model_type == 'classif':
		if kwargs.get('n_classes') == 2:
			loss = torch.nn.BCELoss()
		else:
			loss = torch.nn.NLLLoss()
		# loss = torch.nn.CrossEntropyLoss() # @ todo
		# loss = torch.nn.BCEWithLogitsLoss()  # @ todo: see
		# https://discuss.pytorch.org/t/bce-loss-giving-negative-values/62309/2
		# (In general, you will prefer BCEWithLogitsLoss over BCELoss.)
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')

	# Initialize the optimizer:
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr=lr,
		weight_decay=weight_decay
	)

	# # Initialize the early stopping object:
	# early_stopping = EarlyStopping(
	# 	patience=patience,
	# 	verbose=False,
	# 	path='checkpoint.pt'
	# )

	history = {'train_loss': [], 'valid_loss': []}
	# train the model:
	for epoch in range(n_epochs):
		train_loss = train_epoch(
			train_loader, model, optimizer, loss, device, return_embeddings
		)
		if valid_loader is not None:
			valid_loss = test_epoch(valid_loader, model, loss)
		# early_stopping(valid_loss, model)
		# if early_stopping.early_stop:
		# 	break
		history['train_loss'].append(train_loss)
		if valid_loader is not None:
			history['valid_loss'].append(valid_loss)
		if verbose and epoch % print_interval == 0:
			if valid_loader is not None:
				print(
					'Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
						epoch, train_loss, valid_loss
					)
				)
			else:
				print(
					'Epoch: {} \tTraining Loss: {:.6f}'.format(
						epoch, train_loss
					)
				)
	# model.load_state_dict(torch.load('checkpoint.pt'))
	# model.eval()

	# Plot performance w.r.t. epochs.
	if plot_loss:
		import datetime
		current_datetime = datetime.datetime.now()
		formatted_datetime = current_datetime.strftime("%Y%m%d%H%M%S%f")[:-3]
		# figure_name = '../figures/' + path_kw + '/perfs_vs_epochs.t' + str(trial_index)
		name_kw = kwargs.get('ds_name') + '.' + kwargs.get('model_name') + '.' + kwargs.get('params_idx')
		figure_name = '../figures/gcl_learn/perfs_vs_epochs.' + name_kw + '.' + formatted_datetime
		title = name_kw
		from gcl_frame.figures.gnn_plot import plot_epoch_curves
		plot_epoch_curves(
			history, figure_name, show_fig=False, loss_only=True, title=None
		)

	return model, history


def evaluate_parameters(
		dataset_app, y_app, params, kf, estimator, model_type, device,
		n_epochs=500,
		verbose=True, **kwargs
):
	perf_valid_list = []
	# for each inner CV fold:
	for train_index, valid_index in kf.split(dataset_app, y_app):
		# split the dataset into train and validation sets:
		dataset_train = dataset_app[train_index]
		train_loader = DataLoader(
			dataset_train, params['batch_size'], shuffle=False
		)
		dataset_valid = dataset_app[valid_index]
		valid_loader = DataLoader(
			dataset_valid, params['batch_size'], shuffle=False
		)

		# Train the model:
		model, history = fit_model(
			train_loader,
			estimator,
			params,
			model_type,
			n_epochs,
			device,
			valid_loader=None,  # @TODO: add validation set for early stopping
			verbose=verbose,
			plot_loss=True,
			**kwargs
		)

		# Predict the validation set:
		metric = ('rmse' if model_type == 'reg' else 'accuracy')
		perf_valid, y_pred_valid, y_true_valid = predict(
			valid_loader, model, metric, device,
			model_type=model_type, **kwargs
		)
		perf_valid_list.append(perf_valid)

	# Average the performance over the inner CV folds:
	perf_valid = np.mean(perf_valid_list)
	return perf_valid


def model_selection_for_gnn(
		G_app, y_app, G_test, y_test,
		estimator,
		param_grid,
		model_type,
		fit_test=False,
		n_epochs=500,  # @TODO: make this a parameter.
		parallel=False,
		n_jobs=multiprocessing.cpu_count(),
		read_gm_from_file=False,
		verbose=True,
		**kwargs
):
	# Scale the targets:
	# @TODO: it is better to fit scaler by training set rather than app set.
	# @TODO: use minmax or log instead?
	if model_type == 'reg':
		from sklearn.preprocessing import StandardScaler
		y_scaler = StandardScaler().fit(np.reshape(y_app, (-1, 1)))
		y_app = y_scaler.transform(np.reshape(y_app, (-1, 1)))
		y_test = y_scaler.transform(np.reshape(y_test, (-1, 1)))
	elif model_type == 'classif':
		# ensure that the labels are in the range [0, n_classes - 1].
		# This is important for classification with Sigmoid and BCELoss.
		# If the labels are not in this range, the loss might be negative.
		from sklearn.preprocessing import LabelEncoder
		y_scaler = LabelEncoder().fit(y_app)
		y_app = y_scaler.transform(y_app)
		y_test = y_scaler.transform(y_test)
		# Ensure the values are floats:
		y_app = y_app.astype(float)
		y_test = y_test.astype(float)   # @TODO: is this necessary?
	else:
		raise ValueError('"mode" must be either "reg" or "classif".')
	kwargs['y_scaler'] = y_scaler
	y_app, y_test = np.ravel(y_app), np.ravel(y_test)

	# Convert NetworkX graphs to PyTorch-Geometric compatible dataset:
	from gcl_frame.dataset.nn import NetworkXGraphDataset
	dataset = NetworkXGraphDataset(
		G_app + G_test, np.concatenate((y_app, y_test)),
		node_label_names=kwargs.get('node_labels'),
		edge_label_names=kwargs.get('edge_labels'),
		node_attr_names=kwargs.get('node_attrs'),
		edge_attr_names=kwargs.get('edge_attrs'),
	)
	dataset_app = dataset[:len(G_app)]
	dataset_test = dataset[len(G_app):]

	# @TODO: change it back.
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # torch.device('cpu') #
	if verbose:
		print('device:', device)

	# Set cross-validation method:
	if model_type == 'reg':
		kf = KFold(n_splits=5, shuffle=True, random_state=42)
	elif model_type == 'classif':
		kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	else:
		raise ValueError('"model_type" must be either "reg" or "classif".')

	# Do cross-validation:
	param_list = list(ParameterGrid(param_grid))

	perf_valid_best = (np.inf if model_type == 'reg' else -np.inf)
	for idx, params in get_iters(
			enumerate(param_list),  # @TODO: remove the [0:2]
			desc='model selection for the graph kernel',
			file=sys.stdout,
			length=len(param_list),
			verbose=True
	):
		if verbose:
			print()
			print(params)

		perf_valid = evaluate_parameters(
			dataset_app, y_app, params, kf, estimator, model_type, device,
			n_epochs=n_epochs,
			verbose=verbose,
			params_idx=str(idx),
			**kwargs
		)

		# Update the best parameters:
		if check_if_valid_better(perf_valid, perf_valid_best, model_type):
			perf_valid_best = perf_valid
			params_best = params

	# Refit the best model on the whole dataset:
	metric = ('rmse' if model_type == 'reg' else 'accuracy')
	app_loader = DataLoader(
		dataset_app, params_best['batch_size'], shuffle=False
	)
	model, history = fit_model(
		app_loader, estimator, params_best, model_type, n_epochs, device,
		valid_loader=None, return_embeddings=(not fit_test), verbose=verbose,
		plot_loss=True, params_idx='refit',
		**kwargs
	)
	pred_results = predict(
		app_loader, model, metric, device,
		model_type=model_type,
		return_embeddings=(not fit_test),
		**kwargs
	)

	# Predict the test set:
	if fit_test:
		perf_app, y_pred_app, y_true_app = pred_results
		test_loader = DataLoader(
			dataset_test, params_best['batch_size'], shuffle=False
		)
		perf_test, y_pred_test, y_true_test = predict(
			test_loader, model, metric, device,
			model_type=model_type,
			return_embeddings=False,
			**kwargs
		)
		embed_app = y_pred_app
		embed_test = y_pred_test
	else:
		perf_app, y_pred_app, y_true_app, embed_app = pred_results
		embed_test = None
		perf_test = None

	# Print out the best performance:
	if verbose:
		print('Best app performance: ', perf_app)
		print('Best test performance: ', perf_test)
		print('Best params Gram: ', params_best)

	# Return the best model:
	return model, perf_app, perf_test, embed_app, embed_test, params_best


def check_if_valid_better(perf_valid, perf_valid_best, mode):
	if mode == 'reg':
		return perf_valid < perf_valid_best
	elif mode == 'classif':
		return perf_valid > perf_valid_best
	else:
		raise ValueError('"mode" must be either "reg" or "classif".')
