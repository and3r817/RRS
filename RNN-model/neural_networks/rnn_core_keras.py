# -*- coding: utf-8 -*-
from __future__ import print_function

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)


import numpy as np
from importlib import reload
from tensorflow.python.keras import backend as be
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import RNN, GRU, LSTM, Dense, Activation, Bidirectional, Masking, Embedding
from .rnn_base import RNNBase
import os
from os import environ
from helpers import evaluation


class RNNCore(RNNBase):

    def __init__(self, updater=None, recurrent_layer=None, backend='tensorflow', mem_frac=None,
                 regularization=0.0, **kwargs):
        super(RNNCore, self).__init__(**kwargs)

        self.regularization = regularization
        self.backend = backend
        self.updater = updater
        self.recurrent_layer = recurrent_layer
        self.tf_mem_frac = mem_frac

        self.name = "RNN with categorical cross entropy"

        self.set_keras_backend(self.backend)


    def set_keras_backend(self, backend):
        if be.backend() != backend:
            environ['KERAS_BACKEND'] = backend
            reload(be)
            assert be.backend() == backend

    def _get_model_filename(self, start_time):
        """Return the name of the file to save the current model
        """
        filename = "rnn_cce_" + self._common_filename(start_time) + ".hdf5"
        return filename

    def prepare_networks(self, n_items):

        self.n_items = n_items

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.compat.v1.Session(config=config)

        self.model = Sequential()

### 1. INPUT MODULE
        if self.recurrent_layer.embedding_size > 0:
         # with embedding layer
            # choose the embedding method
            # without pre-trained embedding, learn own embedding through the training process
            if self.recurrent_layer.embedding_method == 'own':
                self.model.add(Embedding(self.n_items, self.recurrent_layer.embedding_size, input_length=self.max_length, trainable=True))
            # using pre-trained lstm-hidden-unit embedding
            elif self.recurrent_layer.embedding_method == 'lstm':
                path = os.getcwd() + '/ks-cooks-1y/embedding/'
                filename = "recipe_%s_emb%d.csv" % (self.recurrent_layer.embedding_method, self.recurrent_layer.embedding_size)
                embedding_matrix = np.genfromtxt(path + filename, delimiter=',')
                self.model.add(
                    Embedding(self.n_items, embedding_matrix.shape[1], weights=[embedding_matrix], mask_zero=True,
                              input_length=self.max_length, trainable=False))
            # using pre-trained tf-idf embedding
            elif self.recurrent_layer.embedding_method == 'tfidf':
                path = os.getcwd() + '/ks-cooks-1y/embedding/'
                filename = "recipe_%s_emb%d.csv" % (self.recurrent_layer.embedding_method, self.recurrent_layer.embedding_size)
                embedding_matrix = np.genfromtxt(path + filename, delimiter=',')
                self.model.add(
                    Embedding(self.n_items, embedding_matrix.shape[1], weights=[embedding_matrix], mask_zero=True,
                              input_length=self.max_length, trainable=False))
            self.model.add(Masking(mask_value=0.0))
        # without embedding layer
        else:
            self.model.add(Masking(mask_value=0.0, input_shape=(self.max_length, self.n_items)))
### 2. RECURRENT MODULE
        rnn = self.get_rnn_type(self.recurrent_layer.layer_type, self.recurrent_layer.bidirectional)
        # define the stacked LSTM layer
        for i, h in enumerate(self.recurrent_layer.layers):
            if i != len(self.recurrent_layer.layers) - 1:
                self.model.add(rnn(h, return_sequences=True, activation=self.active_f))
            else:  # last rnn return only last output
                self.model.add(rnn(h, return_sequences=False, activation=self.active_f))
### 3. OUTPUT MODULE
        self.model.add(Dense(self.n_items))
        self.model.add(Activation('softmax'))
### 4. Loss Function and Optimizer
        optimizer = self.updater()

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def get_rnn_type(self, rnn_type, bidirectional):

        if rnn_type == 'GRU':
            rnn = GRU
        elif rnn_type == 'LSTM':
            rnn = LSTM
        else:
            rnn = RNN

        if bidirectional:
            return Bidirectional(rnn)
        else:
            return rnn

    def _prepare_input(self, sequences):
        """ Sequences is a list of [user_id, input_sequence, targets]
        """
        # print("_prepare_input()")
        batch_size = len(sequences)

        # Shape of return variables
        if self.recurrent_layer.embedding_size > 0:
            X = np.zeros((batch_size, self.max_length),
                         dtype=self._input_type)  # keras embedding requires movie-id sequence, not one-hot
        else:
            X = np.zeros((batch_size, self.max_length, self.n_items), dtype=self._input_type)  # input of the RNN
        Y = np.zeros((batch_size, self.n_items), dtype='float32')  # output target

        for i, sequence in enumerate(sequences):
            user_id, in_seq, target = sequence

            if self.recurrent_layer.embedding_size > 0:
                X[i, :len(in_seq)] = np.array([item[0] for item in in_seq])
            else:
                seq_features = np.array(list(map(lambda x: self._get_features(x), in_seq)))
                X[i, :len(in_seq), :] = seq_features  # Copy sequences into X

            Y[i][target[0]] = 1.

        return X, Y

    def _compute_validation_metrics(self, dataset, metrics):
        """
        add value to lists in v_metrics dictionary
        """
        self.dataset = dataset
        ev = evaluation.Evaluator(self.dataset, k=10)
        # for batch_input, goal in gen_mini_batch(dataset.validation_set(epochs=1)):  # test=True
        for sequence, user_id in self.dataset.validation_set(epochs=1):
            sequence = sequence[-min(self.max_length, len(sequence)):]
            num_viewed = int(len(sequence) / 2)
            viewed = sequence[:num_viewed]
            goal = [i[0] for i in sequence[num_viewed:]]  # list of movie ids

            X = np.zeros((1, self.max_length), dtype=np.int32)  # ktf embedding requires movie-id sequence, not one-hot
            X[0, :len(viewed)] = np.array([item[0] for item in viewed])

            output = self.model.predict_on_batch(X)
            # output[[i[0] for i in viewed]] = -np.inf
            predictions = np.argpartition(-output, list(range(10)), axis=-1)[0, :10]
            # print("predictions")
            # print(predictions)
            ev.add_instance(goal, predictions)

        #
        metrics['recall'].append(ev.average_recall())
        metrics['sps'].append(ev.sps())
        metrics['sps_short'].append(ev.sps_short())
        metrics['sps_long'].append(ev.sps_long())
        metrics['precision'].append(ev.average_precision())
        metrics['ndcg'].append(ev.average_ndcg())
        metrics['user_coverage'].append(ev.user_coverage())
        metrics['item_coverage'].append(ev.item_coverage())
        metrics['total_item_coverage'].append(ev.total_item_coverage())
        metrics['uniq_rec'].append(ev.uniq_rec())
        metrics['blockbuster_share'].append(ev.blockbuster_share())
        metrics['intra_list_similarity'].append(ev.average_intra_list_similarity())

        # del ev
        ev.nb_of_dp = self.dataset.n_items
        ev.instances = []

        return metrics

    def _save(self, filename):
        super(RNNOneHotK, self)._save(filename)
        self.model.save(filename)

    def _load(self, filename):
        '''Load parameters values from a file
        '''
        self.model = load_model(filename)
