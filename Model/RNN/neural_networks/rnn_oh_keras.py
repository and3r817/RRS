# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
from importlib import reload
from keras import backend as be
from keras.models import Sequential, load_model, Model
from keras.layers import RNN, GRU, LSTM, Dense, Activation, Bidirectional, Masking, Embedding
from .rnn_base import RNNBase
from os import environ
from helpers import evaluation


class RNNOneHotK(RNNBase):
    """RNNOneHot are recurrent neural networks that do not depend on the factorization: they are based on one-hot encoding.

The parameters specific to the RNNOneHot are:
diversity_bias: a float in [0, inf) that tunes how the cost function of the network is biased towards less seen movies.
In practice, the classification error given by the categorical cross-entropy is divided by exp(diversity_bias * popularity (on a scale from 1 to 10)).
This will reduce the error associated to movies with a lot of views, putting therefore more importance on the ability of the network to correctly predict the rare movies.
A diversity_bias of 0 produces the normal behavior, with no bias.
    """

    def __init__(self, updater=None, recurrent_layer=None, backend='tensorflow', mem_frac=None, diversity_bias=0.0,
                 regularization=0.0, **kwargs):
        super(RNNOneHotK, self).__init__(**kwargs)

        # self.diversity_bias = np.cast[theano.config.floatX](diversity_bias)

        self.regularization = regularization
        self.backend = backend
        self.updater = updater
        self.recurrent_layer = recurrent_layer
        self.tf_mem_frac = mem_frac

        self.name = "RNN with categorical cross entropy"

        self.set_keras_backend(self.backend)

        if self.backend == 'tensorflow':
            self.framework = 'ktf'
        else:
            self.framework = 'kth'

    def set_keras_backend(self, backend):
        if be.backend() != backend:
            environ['KERAS_BACKEND'] = backend
            reload(be)
            assert be.backend() == backend

    def _get_model_filename(self, epochs):
        """Return the name of the file to save the current model
        """
        # filename = "rnn_cce_db"+str(self.diversity_bias)+"_r"+str(self.regularization)+"_"+self._common_filename(epochs)
        filename = "rnn_cce_" + self._common_filename(epochs) + "." + self.framework
        return filename

    def prepare_networks(self, n_items):

        self.n_items = n_items

        if be.backend() == 'tensorflow':
            import tensorflow as tf
            from keras.backend.tensorflow_backend import set_session
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.tf_mem_frac
            set_session(tf.Session(config=config))

        self.model = Sequential()
        if self.recurrent_layer.embedding_size > 0:
            # embedding_matrix = np.genfromtxt('/Users/xun/Documents/Thesis/Improving-RNN-recommendation-model/Dataset/ks-cooks-1y/recipe_embedding.csv', delimiter=',')
            # self.model.add(
            #     Embedding(self.n_items, embedding_matrix.shape[1], weights=[embedding_matrix], mask_zero=True,
            #               input_length=self.max_length, trainable=False))
            self.model.add(Embedding(self.n_items, self.recurrent_layer.embedding_size, input_length=self.max_length))
            self.model.add(Masking(mask_value=0.0))
        else:
            self.model.add(Masking(mask_value=0.0, input_shape=(self.max_length, self.n_items)))

        rnn = self.get_rnn_type(self.recurrent_layer.layer_type, self.recurrent_layer.bidirectional)

        for i, h in enumerate(self.recurrent_layer.layers):
            if i != len(self.recurrent_layer.layers) - 1:
                self.model.add(rnn(h, return_sequences=True, activation=self.active_f))
            else:  # last rnn return only last output
                self.model.add(rnn(h, return_sequences=False, activation=self.active_f))
        self.model.add(Dense(self.n_items))
        self.model.add(Activation('softmax'))

        optimizer = self.updater()

        # def bpr_loss(y_true, y_pred):
        #     diff = tf.expand_dims(y_pred[0], 1) - y_pred[0]
        #     sig = be.sigmoid(diff)
        #     # sig = be.square(diff)
        #     log_loss = - be.log(sig)
        #     log_loss_mean = be.mean(log_loss, axis=1)
        #     # print(log_loss_mean.shape, y_true.shape)
        #     loss = tf.tensordot(y_true, log_loss_mean, axes=1)
        #     return loss

        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        # self.model.compile(loss=bpr_loss, optimizer=optimizer)

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

            Y[i][target[0][0]] = 1.

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
        metrics['precision'].append(ev.average_precision())
        metrics['ndcg'].append(ev.average_ndcg())
        metrics['user_coverage'].append(ev.user_coverage())
        metrics['item_coverage'].append(ev.item_coverage())
        metrics['blockbuster_share'].append(ev.blockbuster_share())

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
