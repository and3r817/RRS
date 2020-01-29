from __future__ import print_function

from numpy.random import seed
seed(1)

from helpers.data_handling import DataHandler

import numpy as np
import random
import pickle


def gen_mini_batch(n_users, max_length, sequence_generator, test=False):
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
    uid = []
    sequences = []
    for user in range(n_users):
        sequence, user_id = next(sequence_generator)
        uid.append(user_id)

        # finds the lengths of the different subsequences
        if not test:  # training set
            seq_lengths = sorted(
                random.sample(range(2, len(sequence)),
                              len(sequence) - 2) )

        else:  # validating set
            seq_lengths = [int(len(sequence)-1)]  # not iterate

        start_l = []
        skipped_seq = 0
        for l in seq_lengths:
            # only one target
            target = sequence[l:][0]
            if len(target) == 0:
                skipped_seq += 1
                continue
            start = max(0, l - max_length)  # sequences cannot be longer than self.max_length
            start_l.append(start)
            sequences.append([user_id, sequence[start:l], target])
            print(user_id, len(sequence), seq_lengths, start_l)

    return sequences

dataset = DataHandler(dirname="ks-cooks-1y")

n_users = dataset.training_set.n_users
n_val_users = dataset.validation_set.n_users
max_length = 60

train_generator = gen_mini_batch(n_users, max_length, dataset.training_set(max_length=850))
val_generator = gen_mini_batch(n_val_users, max_length, dataset.training_set(max_length=850), test = True)

path = '~/Dataset/'
dirname = 'ks-cooks-1y'

with open(path+dirname+'/data/sub_sequences_all_list.pickle', 'wb') as fp:
    pickle.dump(train_generator, fp)

with open(path+dirname+'/data/validation_all_list.pickle', 'wb') as fp:
    pickle.dump(val_generator, fp)


