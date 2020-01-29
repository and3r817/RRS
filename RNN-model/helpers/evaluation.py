# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from scipy import sparse as ssp
import os.path
import collections


# Plot multiple figures at the same time
# plt.ion()

class Evaluator(object):
    '''Evaluator is a class to compute metrics on tests

    It is used by first adding a series of "instances" : pairs of goals and predictions, then metrics can be computed on the ensemble of instances:
    average precision, percentage of instance with a correct prediction, etc.

    It can also return the set of correct predictions.
    '''

    def __init__(self, dataset, k=10):
        super(Evaluator, self).__init__()
        self.instances = []
        self.dataset = dataset
        self.k = k

        self.metrics = {'sps': self.short_term_prediction_success,
                        'sps_short': self.sps_short,
                        'sps_long': self.sps_long,
                        'recall': self.average_recall,
                        'precision': self.average_precision,
                        'ndcg': self.average_ndcg,
                        'uniq_rec': self.uniq_rec,
                        'total_item_coverage': self.total_item_coverage,
                        'item_coverage': self.item_coverage,
                        'user_coverage': self.user_coverage,
                        'assr': self.assr,
                        'blockbuster_share': self.blockbuster_share,
                        'intra_list_similarity': self.average_intra_list_similarity}

    def add_instance(self, goal, predictions, iter=False):
        if not iter:
            self.instances.append([goal, predictions])
        else:
            # self.instances.append([goal, predictions])
            sps = int(goal[0] in predictions[:min(len(predictions), self.k)])
            recall = float(len(set(goal) & set(predictions[:min(len(predictions), self.k)]))) / len(goal)
            return sps, recall

    def _load_interaction_matrix(self):
        '''Load the training set as an interaction matrix between items and users in a sparse format.
        '''
        filename = self.dataset.dirname + '/data/train_set_triplets'
        if os.path.isfile(filename + '.npy'):
            file_content = np.load(filename + '.npy')
        else:
            file_content = np.loadtxt(filename)
            np.save(filename, file_content)

        self._interactions = ssp.coo_matrix(
            (np.ones(file_content.shape[0]), (file_content[:, 1].astype(int), file_content[:, 0].astype(int)))).tocsr()

    def _intra_list_similarity(self, items):
        '''Compute the intra-list similarity of a list of items.
        '''
        if not hasattr(self, "_interactions"):
            self._load_interaction_matrix()

        norm = np.sqrt(np.asarray(self._interactions[items, :].sum(axis=1)).ravel())
        sims = self._interactions[items, :].dot(self._interactions[items, :].T).toarray()
        S = 0
        for i in range(len(items)):
            for j in range(i):
                S += sims[i, j] / norm[i] / norm[j]

        return S

    def average_intra_list_similarity(self):
        '''Return the average intra-list similarity, as defined in "Auralist: Introducing Serendipity into Music Recommendation"
        '''

        ILS = 0
        for goal, prediction in self.instances:
            if len(prediction) > 0:
                ILS += self._intra_list_similarity(prediction[:min(len(prediction), self.k)])

        return ILS / len(self.instances)

    def blockbuster_share(self):
        '''Return the percentage of correct long term predictions that are about items in the top 1% of the most popular items.
        '''

        correct_predictions = self.get_correct_predictions()
        nb_pop_items = self.dataset.n_items // 100
        pop_items = np.argpartition(-self.dataset.item_popularity, nb_pop_items)[:nb_pop_items]

        if len(correct_predictions) == 0:
            return 0
        return len([i for i in correct_predictions if i in pop_items]) / len(correct_predictions)


    def average_novelty(self):
        '''Return the average novelty measure, as defined in "Auralist: Introducing Serendipity into Music Recommendation"
        '''

        nb_of_ratings = sum(self.dataset.item_popularity)

        novelty = 0
        for goal, prediction in self.instances:
            if len(prediction) > 0:
                novelty += sum(map(np.log2, self.dataset.item_popularity[
                    prediction[:min(len(prediction), self.k)]] / nb_of_ratings)) / min(len(prediction), self.k)

        return -novelty / len(self.instances)

    def average_precision(self):
        '''Return the average number of correct predictions per instance.
        '''
        precision = 0
        i = 0
        for goal, prediction in self.instances:
            if len(prediction) > 0:
                precision += float(
                    len(set(goal) & set(prediction[:min(len(prediction), self.k)]))
                ) / min(len(prediction), self.k)
            # if i < 30:
            # 	print(goal, prediction, precision)
            # 	i += 1
            if min(len(prediction), self.k) < 10:
                print(len(prediction))
        return precision / len(self.instances)

    def average_recall(self):
        '''Return the average recall.
        '''
        recall = 0
        i = 0
        for goal, prediction in self.instances:
            if len(goal) > 0:
                recall += float(len(set(goal) & set(prediction[:min(len(prediction), self.k)]))) / len(goal)
            # if i < 30:
            # 	print(goal, prediction, recall)
            # 	i += 1

        return recall / len(self.instances)

    def average_ndcg(self):
        ndcg = 0.
        for goal, prediction in self.instances:
            if len(prediction) > 0:
                dcg = 0.
                max_dcg = 0.
                for i, p in enumerate(prediction[:min(len(prediction), self.k)]):
                    if i < len(goal):
                        max_dcg += 1. / np.log2(2 + i)

                    if p in goal:
                        dcg += 1. / np.log2(2 + i)

                ndcg += dcg / max_dcg

        return ndcg / len(self.instances)

    def short_term_prediction_success(self):
        '''Return the percentage of instances for which the first goal was in the predictions
        '''
        score = 0
        for goal, prediction in self.instances:
            try:
                score += int(goal[0] in prediction[:min(len(prediction), self.k)])
            # print(goal[0])
            # print(prediction)
            # exit()
            except Exception as e:
                print(e)
                print(goal[0])
                print(prediction)
                exit()

        return score / len(self.instances)

    def sps_short(self):
        '''Return the sps for short sequences
        '''
        short_count = 0
        score = 0
        for goal, prediction in self.instances:
            if len(goal) < 15:
                short_count += 1
                try:
                    score += int(goal[0] in prediction[:min(len(prediction), self.k)])
                # print(goal[0])
                # print(prediction)
                # exit()
                except Exception as e:
                    print(e)
                    print(goal[0])
                    print(prediction)
                    exit()
        return score / short_count


    def sps_long(self):
        '''Return the sps for long sequences
        '''
        long_count = 0
        score = 0
        for goal, prediction in self.instances:
            if len(goal) >= 15:
                long_count += 1
                try:
                    score += int(goal[0] in prediction[:min(len(prediction), self.k)])
                # print(goal[0])
                # print(prediction)
                # exit()
                except Exception as e:
                    print(e)
                    print(goal[0])
                    print(prediction)
                    exit()
        return score / long_count

    def sps(self):
        return self.short_term_prediction_success()

    def user_coverage(self):

        '''Return the percentage of instances for which at least one of the goals was in the predictions
        '''
        score = 0
        for goal, prediction in self.instances:
            score += int(len(set(goal) & set(prediction[:min(len(prediction), self.k)])) > 0)

        return score / len(self.instances)

    def get_all_goals(self):
        '''Return a concatenation of the goals of each instances
        '''
        return [g for goal, _ in self.instances for g in goal]

    def get_strict_goals(self):
        '''Return a concatenation of the strict goals (i.e. the first goal) of each instances
        '''
        return [goal[0] for goal, _ in self.instances]

    def get_all_predictions(self):
        '''Return a concatenation of the predictions of each instances
        '''
        return [p for _, prediction in self.instances for p in prediction[:min(len(prediction), self.k)]]

    def get_correct_predictions(self):
        '''Return a concatenation of the correct predictions of each instances
        '''
        correct_predictions = []
        for goal, prediction in self.instances:
            correct_predictions.extend(list(set(goal) & set(prediction[:min(len(prediction), self.k)])))
        return correct_predictions

    def item_coverage(self):
        return len(set(self.get_correct_predictions()))

    def total_item_coverage(self):
        '''Return the total number of the correct predictions of each instances
        '''
        return len(self.get_correct_predictions())

    def uniq_rec(self):
        '''Return the total number of unique items in the recommendation list
        '''
        rec_l = []
        for goal, prediction in self.instances:
            rec_l.append(prediction)
        rec_item_list = [item for sublist in rec_l for item in sublist]
        return len(set(rec_item_list))

    def get_correct_strict_predictions(self):
        '''Return a concatenation of the strictly correct predictions of each instances (i.e. predicted the first goal)
        '''
        correct_predictions = []
        for goal, prediction in self.instances:
            correct_predictions.extend(list(set([goal[0]]) & set(prediction[:min(len(prediction), self.k)])))
        return correct_predictions

    def get_rank_comparison(self):
        '''Returns a list of tuple of the form (position of the item in the list of goals, position of the item in the recommendations)
        '''
        all_positions = []
        for goal, prediction in self.instances:
            position_in_predictions = np.argsort(prediction)[goal]
            all_positions.extend(list(enumerate(position_in_predictions)))

        return all_positions



