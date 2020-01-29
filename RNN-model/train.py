from __future__ import print_function

from numpy.random import seed
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

import tensorflow as tf
tf.random.set_seed(2)

import numpy as np
import helpers.command_parser as cp
import helpers.command_parser as parse
from helpers.data_handling import DataHandler
import os, datetime



def training_command_parser(parser):
	parser.add_argument('--tshuffle', help='Shuffle sequences during training.', action='store_true')
	parser.add_argument('--extended_set', help='Use extended training set (contains first half of validation and test set).', action='store_true')
	parser.add_argument('-d', dest='dataset', help='Directory name of the dataset.', default='', type=str)
	parser.add_argument('--dir', help='Directory name to save model.', default='', type=str)
	parser.add_argument('--save', choices=['All', 'Best', 'None'], help='Policy for saving models.', default='Best')
	parser.add_argument('--metrics', help='Metrics for validation, comma separated', default='sps', type=str)
	parser.add_argument('--n_epoch', help='Number of epochs', default=20, type=int)


def num(s):
	try:
		return int(s)
	except ValueError:
		return float(s)


def main():

	args = cp.command_parser(training_command_parser, cp.predictor_command_parser)
	predictor = parse.get_predictor(args)

	dataset = DataHandler(dirname=args.dataset, extended_training_set=args.extended_set, shuffle_training=args.tshuffle)
	logdir = os.path.join(os.getcwd(),"logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
	tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

	predictor.prepare_networks(dataset.n_items)

	predictor.train(dataset,
		tensorboard_callback=tensorboard_callback,
		autosave=args.save,
		save_dir= dataset.dirname + "/models/" + args.dir,
		n_epoch=args.n_epoch,
	    validation_metrics = args.metrics.split(','))


if __name__ == '__main__':
	main()
