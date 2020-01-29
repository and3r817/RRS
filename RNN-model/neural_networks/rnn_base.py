# -*- coding: utf-8 -*-
from __future__ import print_function

from numpy.random import seed

seed(1)
import tensorflow as tf

tf.random.set_seed(2)

import glob
import os
import random
import re
import sys
from time import time
import datetime

import numpy as np
import pickle


from tensorflow.keras.callbacks import ModelCheckpoint
from helpers import evaluation


class RNNBase(object):
    """Base for RNN object.
	"""

    def __init__(self,
                 active_f='tanh',
                 max_length=60,
                 batch_size=32):

        super(RNNBase, self).__init__()

        self.max_length = max_length
        self.batch_size = batch_size
        self.active_f = active_f
        self._input_type = 'float32'

        self.name = "RNN base"
        self.metrics = {'recall': {'direction': 1},
                        'precision': {'direction': 1},
                        'sps': {'direction': 1},
                        'sps_short': {'direction': 1},
                        'sps_long': {'direction': 1},
                        'user_coverage': {'direction': 1},
                        'item_coverage': {'direction': 1},
                        'total_item_coverage': {'direction': 1},
                        'uniq_rec': {'direction': 1},
                        'ndcg': {'direction': 1},
                        'blockbuster_share': {'direction': -1},
                        'intra_list_similarity': {'direction': 1}
                        }

    def _common_filename(self, start_time):
        '''Common parts of the filename across sub classes.
		'''
        filename = "ml" + str(self.max_length) + "_bs" + str(self.batch_size) + "_ne" + str(
            start_time) + "_" + self.recurrent_layer.name + "_" + self.updater.name


        if self.active_f != 'tanh':
            filename += "_act" + self.active_f[0].upper()
        return filename

    def top_k_recommendations(self, sequence, k=10, exclude=None):
        ''' Receives a sequence of (id, rating), and produces k recommendations (as a list of ids)
		'''

        seq_by_max_length = sequence[-min(self.max_length, len(sequence)):]  # last max length or all

        # Prepare RNN input
        if self.recurrent_layer.embedding_size > 0:
            X = np.zeros((1, self.max_length), dtype=np.int32)
            X[0, :len(seq_by_max_length)] = np.array([item[0] for item in seq_by_max_length])
        else:
            X = np.zeros((1, self.max_length, self._input_size()), dtype=self._input_type)  # input shape of the RNN
            X[0, :len(seq_by_max_length), :] = np.array(list(map(lambda x: self._get_features(x), seq_by_max_length)))

        # Run RNN
        output = self.model.predict_on_batch(X)

        # filter out viewed items
        output_array = output.numpy()
        output_array[0][[i[0] for i in sequence]] = -np.inf
        output = tf.convert_to_tensor(output_array)

        return list(np.argpartition(-output[0], list(range(k)))[:k])

    def set_dataset(self, dataset):
        self.dataset = dataset

    def train(self, dataset, tensorboard_callback,
              autosave='Best',
              save_dir='',
              n_epoch=10,
              validation_metrics=['sps']):

        self.dataset = dataset

        if len(set(validation_metrics) & set(self.metrics.keys())) < len(validation_metrics):
            raise ValueError(
                'Incorrect validation metrics. Metrics must be chosen among: ' + ', '.join(self.metrics.keys()))

        X, Y = self._gen_mini_batch(self.dataset.dirname)
        x_val, y_val = self._gen_mini_batch(self.dataset.dirname, test=True)

        start_time = time()

        train_costs = []
        metrics = {name: [] for name in self.metrics.keys()}
        filename = {}

        try:
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            filepath = save_dir + self._get_model_filename(timestamp)

            # get rid of ".hdf5" so that we can add the epoch index in the model name
            filepath = filepath[0:-5]

            checkpoint = ModelCheckpoint(filepath + '{epoch:03d}.hdf5', verbose=1,
                                         monitor='val_loss', save_best_only=True, mode='auto')

            history = self.model.fit(X, Y, epochs=n_epoch, batch_size=self.batch_size,
                                     validation_data=(x_val, y_val),
                                     # workers = 1, use_multiprocessing = True,
                                     callbacks=[checkpoint, tensorboard_callback],
                                     verbose=2)
            cost = history.history['loss']
            print(cost)

            # Compute all evaluation metrics for validation set
            metrics = self._compute_validation_metrics(self.dataset, metrics)

            # Print info
            self._print_progress(n_epoch, start_time, train_costs
                                 , metrics, validation_metrics
                                 )

        except KeyboardInterrupt:
            print('Training interrupted')

        best_run = np.argmax(
            np.array(metrics[validation_metrics[0]]) * self.metrics[validation_metrics[0]]['direction'])
        return ({m: metrics[m][best_run] for m in self.metrics.keys()}, time() - start_time, filename[best_run])

    def _gen_mini_batch(self, dirname, test=False):

        sequence_train_all = np.load(dirname + '/data/sub_sequences_all_list.pickle', allow_pickle=True)
        sequence_val_all = np.load(dirname + '/data/validation_all_list.pickle', allow_pickle=True)
        if not test:
            return self._prepare_input(sequence_train_all)
        else:
            return self._prepare_input(sequence_val_all)

    def _print_progress(self, epochs, start_time, train_costs
                        , metrics
                        , validation_metrics
                        ):
        '''Print learning progress in terminal
		'''
        print(self.name, epochs, " epochs in", time() - start_time, "s")
        print("Last train cost : ", train_costs[-1])
        for m in self.metrics:
            print(m, ': ', metrics[m][-1])
            if m in validation_metrics:
                print('Best ', m, ': ',
                      max(np.array(metrics[m]) * self.metrics[m]['direction']) * self.metrics[m]['direction'])

        print('-----------------')

    def _save(self, filename):
        '''Save the parameters of a network into a file
		'''
        print('Save model in ' + filename)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    def _input_size(self):
        ''' Returns the number of input neurons
		'''
        return self.n_items

    def _get_features(self, item):
        '''Change a tuple (item_id, rating) into a list of features to feed into the RNN
		features have the following structure: [one_hot_encoding, personal_rating on a scale of ten, average_rating on a scale of ten, popularity on a log scale of ten]
		'''

        one_hot_encoding = np.zeros(self.n_items)
        one_hot_encoding[item[0]] = 1
        return one_hot_encoding
