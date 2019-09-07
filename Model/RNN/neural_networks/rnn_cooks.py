from __future__ import print_function

from helpers import evaluation
from helpers.data_handling import DataHandler

import numpy as np
import random
import os
from time import time

from keras import backend as be
from keras.models import Sequential, load_model, Model
from keras.layers import RNN, GRU, LSTM, Dense, Activation, Bidirectional, Masking, Embedding
from keras.optimizers import Adagrad, Adam, SGD, RMSprop

n_items = 1477
embedding_size = 8
max_length = 50
layers = [50]  # one LSTM layer with 50 hidden neurons
active_f = 'tanh'  # activation for rnn
batch_size = 64

learning_rate = 0.1
input_type = 'float32'

metrics = {'recall': {'direction': 1},
           'precision': {'direction': 1},
           'sps': {'direction': 1},
           'user_coverage': {'direction': 1},
           'item_coverage': {'direction': 1},
           'ndcg': {'direction': 1},
           'blockbuster_share': {'direction': -1}
           }


# class RNNOneHotK(object):

def prepare_networks(n_items, embedding_size, max_length):
    if be.backend() == 'tensorflow':
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = self.tf_mem_frac
        set_session(tf.Session(config=config))

        model = Sequential()
        if embedding_size > 0:
            model.add(Embedding(n_items, embedding_size, input_length=max_length))
            model.add(Masking(mask_value=0.0))
        else:
            model.add(Masking(mask_value=0.0, input_shape=(max_length, n_items)))

    rnn = LSTM

    for i, h in enumerate(layers):
        if i != len(layers) - 1:
            model.add(rnn(h, return_sequences=True, activation=active_f))
        else:  # last rnn return only last output
            model.add(rnn(h, return_sequences=False, activation=active_f))
    model.add(Dense(n_items))
    model.add(Activation('softmax'))

    # optimizer = self.updater()
    def gpu_diag_wide(X):
        E = be.eye(*X.shape)
        return be.sum(X * E, axis=1)

    def bpr(yhat):
        return be.mean(-be.log(be.sigmoid(tf.expand_dims(gpu_diag_wide(yhat), 1) - yhat)))

    def identity_loss(y_true, y_pred):
        return y_true, y_pred

    def customLoss(y_true, y_pred):
        # target_index = be.argmax(y_true)
        target_index = np.argmax(y_true)
        # target_index = be.get_value(target_index)
        y_true_pred = y_pred[0][target_index]
        y_true_pred = be.reshape(y_true_pred, [1, ])
        y_true_vec = tf.expand_dims(y_true_pred, 1)
        # r_uij = y_true_vec - be.transpose(y_pred)
        r_uij = y_true_vec - y_pred
        return be.mean(-be.log(be.sigmoid(r_uij)))
    
    # model.compile(loss='categorical_crossentropy', optimizer=Adagrad(lr=learning_rate))
    model.compile(loss=customLoss, optimizer=Adagrad(lr=learning_rate))
    # model.layers[-1].output
    # print(model.layers[-1].output.shape)
    # model.compile(loss=bpr(model.layers[-1].output), optimizer=Adagrad(lr=learning_rate))
    return model


def get_features(item_id, n_items):
    '''Change a tuple (item_id, rating) into a list of features to feed into the RNN
    features have the following structure: [one_hot_encoding, personal_rating on a scale of ten, average_rating on a scale of ten, popularity on a log scale of ten]
    '''

    one_hot_encoding = np.zeros(n_items)
    one_hot_encoding[item_id] = 1
    return one_hot_encoding


def _input_size(dataset):
    ''' Returns the number of input neurons
    '''
    return dataset.n_items


def save(filename):
    '''Save the parameters of a network into a file
    '''
    print('Save model in ' + filename)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def _common_filename(epochs, batch_size = batch_size):
    '''Common parts of the filename accros sub classes.
    '''
    filename = "ml" + str(max_length) + "_bs" + str(batch_size) + "_ne" + str(
        epochs)
    return filename



def _get_model_filename(epochs):
    """Return the name of the file to save the current model
    """
    # filename = "rnn_cce_db"+str(self.diversity_bias)+"_r"+str(self.regularization)+"_"+self._common_filename(epochs)
    filename = "rnn_cce_" + _common_filename(epochs)
    return filename


def prepare_input(sequences, max_length=max_length, embedding_size=embedding_size):
    """ Sequences is a list of [user_id, input_sequence, targets]
    """
    # print("_prepare_input()")
    batch_size = len(sequences)
    # print(batch_size)

    # Shape of return variables
    if embedding_size > 0:
        X = np.zeros((batch_size, max_length),
                     dtype='int')  # keras embedding requires movie-id sequence, not one-hot
    else:
        X = np.zeros((batch_size, max_length, n_items), dtype='int')  # input of the RNN
    Y = np.zeros((batch_size, n_items), dtype='int')  # output target

    for i, sequence in enumerate(sequences):
        user_id, in_seq, target = sequence

        if embedding_size > 0:
            X[i, :len(in_seq)] = np.array([item[0] for item in in_seq])
        else:
            seq_features = np.array(list(map(lambda x: _get_features(x, n_items), in_seq)))
            X[i, :len(in_seq), :] = seq_features  # Copy sequences into X

        Y[i][target[0]] = 1.
    return X, Y


def gen_mini_batch(sequence_generator, batch_size=64, iter=False, test=False, max_reuse_sequence=3):
    ''' Takes a sequence generator and produce a mini batch generator.
    The mini batch have a size defined by self.batch_size, and have format of the input layer of the rnn.

    test determines how the sequence is splitted between training and testing
        test == False, the sequence is split randomly
        test == True, the sequence is split in the middle

    if test == False, max_reuse_sequence determines how many time a single sequence is used in the same batch.
        with max_reuse_sequence = inf, one sequence will be used to make the whole batch (if the sequence is long enough)
        with max_reuse_sequence = 1, each sequence is used only once in the batch
    N.B. if test == True, max_reuse_sequence = 1 is used anyway
    '''
    # iterations = 0
    while True:
        j = 0
        sequences = []
        batch_size = batch_size
        if test:
            batch_size = 1
        while j < batch_size:  # j : # of precessed sequences for rnn input

            sequence, user_id = next(sequence_generator)

            # finds the lengths of the different subsequences
            if not test:  # training set
                seq_lengths = sorted(
                    random.sample(range(5, min(len(sequence), max_length)),  # range, min_length = 5
                                  min([batch_size - j, max_reuse_sequence]))  # always only take 3 subsequence from the sequences
                )
                # print(seq_lengths)
            elif iter:
                batch_size = len(sequence) - 1
                seq_lengths = list(range(1, len(sequence)))
            else:  # validating set
                seq_lengths = [int(len(sequence) / 2)]  # half of len

            skipped_seq = 0
            for l in seq_lengths:
                # target is only for rnn with hinge, logit and logsig.
                start = np.random.randint(0, len(sequence)) # randomly choose a start position
                start = min(start, len(sequence)-l-1)

                target = sequence[start + l:][0]

                if len(target) == 0:
                    skipped_seq += 1
                    continue
                # start = max(0, l - max_length)  # sequences cannot be longer than self.max_length
                # print(target)
                sequences.append([user_id, sequence[start:start + l], target])
            # print([user_id, sequence[start:l], target])

            j += len(seq_lengths) - skipped_seq
        # print(j, len(sequences), sequences[0])
        # iterations += 1
        # print('generating mini_batch ({})'.format(iterations))
        if test:
            yield prepare_input(sequences), [i[0] for i in sequence[seq_lengths[0]:]]
            # return prepare_input(sequences), [i[0] for i in sequence[seq_lengths[0]:]]
        else:
            yield prepare_input(sequences)
            # return prepare_input(sequences)


def compute_validation_metrics(model, dataset, metrics = metrics):
    """
    add value to lists in metrics dictionary
    """

    ev = evaluation.Evaluator(dataset, k=10)

    # for batch_input, goal in gen_mini_batch(dataset.validation_set(epochs=1)):  # test=True
    for sequence, user_id in dataset.validation_set(epochs=1):
        sequence = sequence[-min(max_length, len(sequence)):]
        num_viewed = int(len(sequence) / 2)
        viewed = sequence[:num_viewed]
        goal = [i[0] for i in sequence[num_viewed:]]  # list of movie ids
        # print(batch_input[0].shape())
        # output = model.predict_on_batch(batch_input[0])
        X = np.zeros((1, max_length), dtype=np.int32)  # ktf embedding requires movie-id sequence, not one-hot
        X[0, :len(viewed)] = np.array([item[0] for item in viewed])

        output = model.predict_on_batch(X)
        predictions = np.argpartition(-output, list(range(10)), axis=-1)[0, :10]
        # print("predictions")
        # print(predictions)
        ev.add_instance(goal, predictions)

    #
    # metrics['recall'].append(ev.average_recall())
    # metrics['sps'].append(ev.sps())
    # metrics['precision'].append(ev.average_precision())
    # metrics['ndcg'].append(ev.average_ndcg())
    # metrics['user_coverage'].append(ev.user_coverage())
    # metrics['item_coverage'].append(ev.item_coverage())
    # metrics['blockbuster_share'].append(ev.blockbuster_share())

    # del ev
    ev.nb_of_dp = dataset.n_items
    v_result = ev.sps()
    ev.instances = []

    return v_result


def train(model,
          # batch_generator,
          dataset,
          max_time=np.inf,
          # progress=2000,
          progress=2,
          autosave='All',
          save_dir='/Users/xun/Documents/Thesis/Improving-RNN-recommendation-model/Dataset/ks-cooks-1y/models',
          # min_iterations=10000,
          min_iterations=2,
          # max_iter=30000,
          max_iter=4,
          load_last_model=False,
          early_stopping=None,
          validation_metrics='sps'):

    # Load last model if needed
    iterations = 0
    epochs_offset = 0


    batch_generator = gen_mini_batch(dataset.training_set(max_length=800), batch_size = 1)

    start_time = time()
    next_save = int(progress)
    # val_costs = []
    train_costs = []
    current_train_cost = []
    epochs = []
    filename = {}



    try:
        while time() - start_time < max_time and iterations < max_iter:
            # Train with a new batch
            try:
                batch = next(batch_generator)

                # self.model.fit(batch[0], batch[2])
                cost = model.train_on_batch(batch[0], batch[1])


 ###################### Here is the part to check intermediet result#################################

                # intermediate_model = Model(inputs=model.layers[0].input,
                #                            outputs=[l.output for l in model.layers[1:]])
                #
                # intermediate_output = intermediate_model.predict(batch[0])
                # print(intermediate_output)
                #
                # outputs = model.predict_on_batch(batch[0])
                # print(outputs[0, :6])
                # print(batch[1])

                if np.isnan(cost):
                    raise ValueError("Cost is NaN")
            except StopIteration:
                break

            current_train_cost.append(cost)
            # current_train_cost.append(0)

            # Check if it is time to save the model
            iterations += 1

            if iterations >= next_save:
                if iterations >= min_iterations:
                    epochs.append(epochs_offset + dataset.training_set.epochs)

                    # Average train cost per epochs after n iterations, n = next_save
                    train_costs.append(np.mean(current_train_cost))
                    current_train_cost = []

                    # Compute validation cost
                    v_metrics = compute_validation_metrics(model, dataset)

                    # Print info
                    print("iteration: ", iterations, "epochs[-1]:", epochs[-1], "train_cost:", train_costs,
                          validation_metrics, v_metrics)

                    # Save model
                    run_nb = len(train_costs) - 1
                    filename[run_nb] = save_dir + "/" + _get_model_filename(round(epochs[-1], 3))
                    model.save(filename[run_nb])

                next_save += progress
    except KeyboardInterrupt:
        print('Training interrupted')
    return train_costs




def top_k_recommendations(model, sequence, max_length, embedding_size, n_items, k=10):
    ''' Receives a sequence of (id, rating), and produces k recommendations (as a list of ids)'''
    seq_by_max_length = sequence[-min(max_length, len(sequence)):]  # last max length or all
    # Prepare RNN input
    if embedding_size > 0:
        X = np.zeros((1, max_length), dtype=np.int32)  # keras embedding requires movie-id sequence, not one-hot
        X[0, :len(seq_by_max_length)] = np.array([item[0] for item in seq_by_max_length])
    else:
        X = np.zeros((1, max_length, n_items), dtype=np.int32)  # input shape of the RNN
        X[0, :len(seq_by_max_length), :] = np.array(
            list(map(lambda x: get_features(x), seq_by_max_length)))

    # Run RNN

    output = model.predict_on_batch(X)
    # output = model.predict(X, batch_size = 1000)

    # find top k according to output
    return list(np.argpartition(-output[0], list(range(k)))[:k])


def _get_features(n_items, item):
    '''Change a tuple (item_id, rating) into a list of features to feed into the RNN
    features have the following structure: [one_hot_encoding, personal_rating on a scale of ten, average_rating on a scale of ten, popularity on a log scale of ten]
    '''

    one_hot_encoding = np.zeros(n_items)
    one_hot_encoding[item[0]] = 1
    return one_hot_encoding


# """
dataset = DataHandler(dirname="ks-cooks-1y")

model = prepare_networks(dataset.n_items, embedding_size, max_length)

loss = train(model, dataset)
#
# print(loss)

# train_generator = gen_mini_batch(dataset.training_set(max_length=800), batch_size=64) #, batch_size=256
# val_generator = gen_mini_batch(dataset.validation_set())

# result = model.fit_generator(generator=train_generator, epochs=1, steps_per_epoch=60,  # 15100/256
#                              # validation_data=val_generator, validation_steps=1,
#                             verbose=1)
#
# result = model.fit(X, Y, epochs=1, batch_size = 256,
#                              # validation_data=val_generator, validation_steps=1,
#                              verbose=1)
# print(result.history)
#
# model.save('/Users/xun/Documents/Thesis/Improving-RNN-recommendation-model/Dataset/ks-cooks-1y/models/rnn-long-train.h5')

# """
