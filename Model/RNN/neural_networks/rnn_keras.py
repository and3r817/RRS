from __future__ import print_function

import numpy as np
from importlib import reload
from keras import backend as be
from keras.models import Sequential, load_model
from keras.layers import RNN, GRU, LSTM, Dense, Activation, Bidirectional, Masking, Embedding
from os import environ
# from .target_selection import SelectTargets
from helpers import evaluation
from helpers.data_handling_notest import DataHandler
from keras.optimizers import Adagrad, Adam, SGD, RMSprop

import random
import time
import os


n_items = 1486
embedding_size = 8
max_length = 30
layers = [3]
active_f = 'tanh'
learning_rate = 0.01
iter=False
input_type = 'float32'
metrics = {'recall': {'direction': 1},
						'precision': {'direction': 1},
						'sps': {'direction': 1},
						'user_coverage': {'direction': 1},
						'item_coverage': {'direction': 1},
						'ndcg': {'direction': 1},
						'blockbuster_share': {'direction': -1}
						}

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
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learning_rate))

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

def prepare_input(sequences, max_length = max_length, embedding_size = 8):
    """ Sequences is a list of [user_id, input_sequence, targets]
    """
    # print("_prepare_input()")
    batch_size = len(sequences)

    # Shape of return variables
    if embedding_size > 0:
        X = np.zeros((batch_size, max_length), dtype='float32')  # keras embedding requires movie-id sequence, not one-hot
    else:
        X = np.zeros((batch_size, max_length, n_items),dtype='float32')  # input of the RNN
    Y = np.zeros((batch_size, n_items), dtype='float32')  # output target

    for i, sequence in enumerate(sequences):
        user_id, in_seq, target = sequence

        if embedding_size > 0:
            X[i, :len(in_seq)] = np.array([item[0] for item in in_seq])
        else:
            seq_features = np.array(list(map(lambda x: get_features(x, n_items), in_seq)))
            X[i, :len(in_seq), :] = seq_features  # Copy sequences into X

        Y[i][target[0]] = 1.
    return X, Y


def gen_mini_batch(sequence_generator, batch_size = 16, iter=False, test=False):
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

    while True:
        j = 0
        sequences = []
        batch_size = batch_size
        if test:
            batch_size = 1
        while j < batch_size:  # j : user order

            sequence, user_id = next(sequence_generator)

            # finds the lengths of the different subsequences
            if not test:  # training set
                seq_lengths = sorted(
                    random.sample(range(2, len(sequence)),  # range
                                  min([batch_size - j, len(sequence) - 2]))  # population
                )
            elif iter:
                batch_size = len(sequence) - 1
                seq_lengths = list(range(1, len(sequence)))
            else:  # validating set
                seq_lengths = [int(len(sequence) / 2)]  # half of len

            skipped_seq = 0
            for l in seq_lengths:
                # target is only for rnn with hinge, logit and logsig.
                target = sequence[l:][0]
                if len(target) == 0:
                    skipped_seq += 1
                    continue
                start = max(0, l - max_length)  # sequences cannot be longer than self.max_length
                # print(target)
                sequences.append([user_id, sequence[start:l], target])
            # print([user_id, sequence[start:l], target])

            j += len(seq_lengths) - skipped_seq  # ?????????

        if test:
            yield prepare_input(sequences), [i[0] for i in sequence[seq_lengths[0]:]]
        else:
            yield prepare_input(sequences)


def compute_validation_metrics(model, dataset, metrics):
    """
    add value to lists in metrics dictionary
    """

    ev = evaluation.Evaluator(dataset, k=10)
    print(iter)
    if not iter:
        for batch_input, goal in gen_mini_batch(dataset.validation_set(), test=False):  # test=True
            # print(batch_input[0].shape())
            # output = model.predict_on_batch(batch_input[0])
            output = model.predict_on_batch(batch_input)
            predictions = np.argpartition(-output, list(range(10)), axis=-1)[0, :10]
            # print("predictions")
            # print(predictions)
            ev.add_instance(goal, predictions)
    else:
        for sequence, user in dataset.validation_set(epochs=1):
            seq_lengths = list(range(1, len(sequence)))  # 1, 2, 3, ... len(sequence)-1
            for length in seq_lengths:
                X = np.zeros((1, max_length, n_items), dtype=input_type)  # input shape of the RNN

                seq_by_max_length = sequence[max(length - max_length, 0):length]  # last max length or all
                X[0, :len(seq_by_max_length), :] = np.array(map(lambda x: get_features(x), seq_by_max_length))

                output = model.predict_on_batch(X)
                predictions = np.argpartition(-output, list(range(10)), axis=-1)[0, :10]
                # print("predictions")
                # print(predictions)
                goal = sequence[length:][0]
                ev.add_instance(goal, predictions)

    metrics['recall'].append(ev.average_recall())
    metrics['sps'].append(ev.sps())
    metrics['precision'].append(ev.average_precision())
    metrics['ndcg'].append(ev.average_ndcg())
    metrics['user_coverage'].append(ev.user_coverage())
    metrics['item_coverage'].append(ev.item_coverage())
    metrics['blockbuster_share'].append(ev.blockbuster_share())

    # del ev
    ev.instances = []

    return metrics

def train(model, dataset,
          max_time=np.inf,
          progress=5000,
          autosave='All',
          save_dir='/Users/xun/Documents/Thesis/Improving-RNN-recommendation-model/k3-3m/models',
          min_iterations=0,
          max_iter=np.inf,
          load_last_model=False,
          early_stopping=None,
          validation_metrics=['sps']):

    dataset = dataset
    # SelectTargets.set_dataset(dataset)

    # Load last model if needed
    iterations = 0
    epochs_offset = 0
    # if load_last_model:
    #     epochs_offset = self.load_last(save_dir)

    # Make batch generator
    # batch_generator = threaded_generator(self._gen_mini_batch(self.sequence_noise(dataset.training_set())))

    batch_generator = gen_mini_batch(dataset.training_set())

    # start_time = time()
    next_save = int(progress)
    # val_costs = []
    train_costs = []
    current_train_cost = []
    epochs = []
    metrics = {}
    metrics = {name: [] for name in metrics.keys()}
    filename = {}

    try:
        while iterations < max_iter:

            # Train with a new batch
            # try:
            batch = next(batch_generator)

                # self.model.fit(batch[0], batch[2])
            cost = model.train_on_batch(batch[0], batch[1])
                # outputs = model.predict_on_batch(batch[0])
                # print(outputs[0, :6])
                # print(batch[1])

            if np.isnan(cost):
                raise ValueError("Cost is NaN")
            #
            # except StopIteration:
            #     break

            current_train_cost.append(cost)
            # current_train_cost.append(0)

            # Check if it is time to save the model
            iterations += 1

            if iterations >= next_save:
                if iterations >= min_iterations:
                    # Save current epoch
                    epochs.append(epochs_offset + dataset.training_set.epochs)

                    # Average train cost
                    train_costs.append(np.mean(current_train_cost))
                    current_train_cost = []

                    # Compute validation cost
                    metrics = compute_validation_metrics(model, dataset, metrics)

                    # Print info
                    print("iteration: ", iterations, "batchs, ", epochs[-1], "cost:",
                          train_costs, metrics,
                          validation_metrics)

                    # Save model
                    run_nb = len(metrics[list(metrics.keys())[0]]) - 1
                    if autosave == 'All':
                        filename[run_nb] = save_dir + "keras" + "/" + "%d" % round(epochs[-1], 3)
                        save(filename[run_nb])
                    # elif autosave == 'Best':
                    #     pareto_runs = self.get_pareto_front(metrics, validation_metrics)
                    if early_stopping is not None:
                        if all([early_stopping(epochs, metrics[m]) for m in validation_metrics]):
                            break
                next_save += progress

    except KeyboardInterrupt:
        print('Training interrupted')

    best_run = np.argmax(
        np.array(metrics[validation_metrics[0]]) * metrics[validation_metrics[0]]['direction'])




dataset = DataHandler(dirname="ks-cooks/ks-cooks-1y")
input = gen_mini_batch(sequence_generator = dataset.training_set())

# for batch_input, goal in gen_mini_batch(dataset.validation_set(), test=False)
# print(dataset.training_set().head())
model = prepare_networks(dataset.n_items, embedding_size, max_length)
##################################
###run the train progress here####
##################################
'''
autosave = 'All'
iterations=0
epochs_offset = 0
early_stopping=None,
validation_metrics=['sps']
# start_time = time()
next_save = int(2.0)
# val_costs = []
train_costs = []
current_train_cost = []
epochs = []
s_metrics = {'recall': {'direction': 1},
						'precision': {'direction': 1},
						'sps': {'direction': 1},
						'user_coverage': {'direction': 1},
						'item_coverage': {'direction': 1},
						'ndcg': {'direction': 1},
						'blockbuster_share': {'direction': -1}
						}
metrics = {name: [] for name in s_metrics.keys()}
filename = {}
save_dir='/Users/xun/Documents/Thesis/Improving-RNN-recommendation-model/k3-3m/models'
current_train_cost=[]


batch_generator = gen_mini_batch(dataset.training_set())
try:
    while iterations < 100:
        batch = next(batch_generator)
        print(batch)
        cost = model.train_on_batch(batch[0], batch[1])
        print(cost)
        current_train_cost.append(cost)
# current_train_cost.append(0)

# Check if it is time to save the model
        iterations += 1

        if iterations >= next_save:
            if iterations >= 1:
                # Save current epoch
                epochs.append(epochs_offset + dataset.training_set.epochs)

                # Average train cost
                train_costs.append(np.mean(current_train_cost))
                current_train_cost = []

                # Compute validation cost
                metrics = compute_validation_metrics(model, dataset, metrics)
                # Print info
                print("iteration: ", iterations, "batchs, ", epochs[-1], "cost:",
                      train_costs, metrics,
                      validation_metrics)

                # Save model
                run_nb = len(metrics[list(metrics.keys())[0]]) - 1
                if autosave == 'All':
                    filename[run_nb] = save_dir + "keras" + "/" + "%d" % round(epochs[-1], 3)
                    save(filename[run_nb])
                # elif autosave == 'Best':
                #     pareto_runs = self.get_pareto_front(metrics, validation_metrics)
                if early_stopping is not None:
                    if all([early_stopping(epochs, metrics[m]) for m in validation_metrics]):
                        break
            next_save += 2.0

except KeyboardInterrupt:
    print('Training interrupted')
    
    
'''

# for batch_input, goal in gen_mini_batch(dataset.validation_set(epochs=1), test=False):  # test=True
#     print(batch_input[0].shape)
#     output = model.predict_on_batch(batch_input)
#     predictions = np.argpartition(-output, list(range(10)), axis=-1)[0, :10]
#     print(output)
#     print(goal)
# train(model, dataset)
# model.save('/Users/xun/Documents/Thesis/model/ks-3m-rnn.h5')
# print(111)

# batch_generator = gen_mini_batch(dataset.training_set())
# batch = next(batch_generator)
# cost = model.train_on_batch(batch[0], batch[1])


train_generator = gen_mini_batch(dataset.training_set())
val_generator = gen_mini_batch(dataset.validation_set())
result = model.fit_generator(generator = train_generator, steps_per_epoch = 10380, epochs=1,
                             validation_data = val_generator, validation_steps= 1038)
print(result.history)
model.save('/Users/xun/Documents/Thesis/model/ks-3m-v2-rnn.h5')
