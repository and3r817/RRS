# -*- coding: utf-8 -*-
from __future__ import print_function

from numpy.random import seed
# seed(1)

# from tensorflow import set_random_seed
# set_random_seed(2)

import logging
logging.getLogger('tensorflow').disabled = True

import glob
import os
import re
import sys
import time
import numpy as np

import helpers.command_parser as parse
from helpers import evaluation
from helpers.data_handling import DataHandler


def get_file_name(predictor, args):
    return args.dir + re.sub('_ml' + str(args.max_length),
                                                         '_ml' + str(args.training_max_length),
                                                         predictor._get_model_filename(args.index))


def find_models(predictor, dataset, args):
    file = dataset.dirname + "/models/" + get_file_name(predictor, args)
    # print("filename : {}".format(file))
    return file


def save_file_name(predictor, dataset, args):
    if not args.save:
        return None
    else:
        file = re.sub('_ne\*_', '_', dataset.dirname + '/results/' + get_file_name(predictor, args))
        return file


def run_tests(predictor, model_file, dataset, args, get_full_recommendation_list=False, k=10):
    predictor._load(model_file)
    # predictor.load_last(os.path.dirname(model_file) + '/')

    # Prepare evaluator
    evaluator = evaluation.Evaluator(dataset, k=k)

    if get_full_recommendation_list:
        k = dataset.n_items

    nb_of_dp = []
    start = time.process_time()
    for sequence, user_id in dataset.test_set(epochs=1):
        if not args.test_iter:
            num_viewed = int(len(sequence) / 2)
            viewed = sequence[:num_viewed]
            goal = [i[0] for i in sequence[num_viewed:]]  # list of movie ids

            recommendations = predictor.top_k_recommendations(viewed, k=k)
            # print(recommendations)
            evaluator.add_instance(goal, recommendations)

            if len(goal) == 0:
                raise ValueError
        else:
            # seq_lengths = sorted(random.sample(xrange(1, len(sequence)),len(sequence) - 1))
            seq_lengths = list(range(1, len(sequence)))

            for length in seq_lengths:
                viewed = sequence[:length]
                goal = sequence[length:][0]

                recommendations = predictor.top_k_recommendations(viewed, k=k)
                evaluator.add_instance(goal, recommendations)

    end = time.process_time()
    print('Timer: ', end - start)
    if len(nb_of_dp) == 0:
        evaluator.nb_of_dp = dataset.n_items
    else:
        evaluator.nb_of_dp = np.mean(nb_of_dp)
    return evaluator


def print_results(ev, metrics, file=None, n_batches=None, print_full_rank_comparison=False):
    for m in metrics:
        if m not in ev.metrics:
            raise ValueError('Unkown metric: ' + m)

        print(m + '@' + str(ev.k) + ': ', ev.metrics[m]())

    if file != None:
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        with open(file, "a") as f:
            f.write(str(n_batches) + "\n")
            for m in metrics:
                f.write(m + '@10' + ': ' + str(ev.metrics[m]()) + "\n")
        if print_full_rank_comparison:
            with open(file + "_full_rank", "a") as f:
                for data in ev.get_rank_comparison():
                    f.write("\t".join(map(str, data)) + "\n")
    else:
        # print("-\t" + "\t".join(map(str, [ev.metrics[m]() for m in metrics])), file=sys.stderr)
        if print_full_rank_comparison:
            with open(file + "_full_rank", "a") as f:
                for data in ev.get_rank_comparison():
                    f.write("\t".join(map(str, data)) + "\n")


def extract_number_of_epochs(filename, args):
    m = re.search('_ne([0-9]+(\.[0-9]+)?)_', filename)
    return args.number_of_batches


def get_last_tested_batch(filename):
    '''If the output file exist already, it will look at the content of the file and return the last batch that was tested.
	This is used to avoid testing to times the same model.
	'''

    if filename is not None and os.path.isfile(filename):
        with open(filename) as f:
            for line in f:
                pass
            return float(line.split()[0])
    else:
        return 0


def test_command_parser(parser):
    parser.add_argument('-d', dest='dataset', help='Directory name of the dataset.', default='', type=str)
    parser.add_argument('-i', dest='index',help='model index', type=int)
    parser.add_argument('-k', dest='nb_of_predictions',
                        help='Number of predictions to make. It is the "k" in "prec@k", "rec@k", etc.', default=10,
                        type=int)
    parser.add_argument('--metrics', help='List of metrics to compute, comma separated',
                        default='sps,sps_short,sps_long,recall,precision,uniq_rec,total_item_coverage,item_coverage,user_coverage,ndcg,blockbuster_share,intra_list_similarity', type=str)
    parser.add_argument('--save', help='Save results to a file', default=False, action='store_true')
    parser.add_argument('--dir', help='Model directory.', default="", type=str)
    parser.add_argument('--save_rank', help='Save the full comparison of goal and prediction ranking.', default='False',
                        action='store_true')
    parser.add_argument('--test_iter', help='test iteratively in every user subsequences', action='store_true')


def main():
    args = parse.command_parser(parse.predictor_command_parser, test_command_parser)

    args.training_max_length = args.max_length
    if args.number_of_batches == -1:
        args.number_of_batches = "*"

    dataset = DataHandler(dirname=args.dataset)
    predictor = parse.get_predictor(args)
    predictor.prepare_networks(dataset.n_items)
    file = find_models(predictor, dataset, args)

    output_file = save_file_name(predictor, dataset, args)

    last_tested_batch = get_last_tested_batch(output_file)
    batches = [extract_number_of_epochs(file, args)]
    file = [file]
    print(file)

    for i, f in enumerate(file):
        if batches[i] > last_tested_batch:
            evaluator = run_tests(predictor, f, dataset, args, k=args.nb_of_predictions)
            print('-------------------')
            print('(', i + 1, '/', len(file), ') results on ' + f)
            print_results(evaluator, args.metrics.split(','), file=output_file, n_batches=batches[i]
                          )


if __name__ == '__main__':
    main()
