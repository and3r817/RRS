import argparse

def command_parser(*sub_command_parser):
    """ *sub_command_parser should be callables that will add arguments to the command parser
	"""

    parser = argparse.ArgumentParser()

    for scp in sub_command_parser:
        scp(parser)

    args = parser.parse_args()
    return args


def predictor_command_parser(parser):
    parser.add_argument('-b', dest='batch_size', help='Batch size', default=64, type=int)
    parser.add_argument('-l', dest='learning_rate', help='Learning rate', default=0.1, type=float)
    parser.add_argument('-r', dest='regularization', help='Regularization (positive for L2, negative for L1)',
                        default=0., type=float)
    parser.add_argument('--loss',
                        help='Loss function, choose between TOP1, BPR and Blackout (Sampling), or hinge, logit and logsig (multi-targets), or CCE (Categorical cross-entropy)',
                        default='CCE', type=str)
    parser.add_argument('--max_length', help='Maximum length of sequences during training (for RNNs)', default=60,
                        type=int)
    parser.add_argument('--act', help='activation function in recurrent layer',
                        choices=['relu', 'elu', 'lrelu', 'tanh'], default='tanh', type=str)
    parser.add_argument('--save_log', help='log history when using tensorflow', action='store_true')
    parser.add_argument('--log_dir', help='Directory name for saving tensorflow log.', default='log', type=str)
    parser.add_argument('--mem_frac', help='memory fraction for tensorflow', default=0.3, type=float)

    from neural_networks.update_manager_k import update_manager_command_parser
    from neural_networks.recurrent_layers import recurrent_layers_command_parser
    recurrent_layers_command_parser(parser)
    update_manager_command_parser(parser)



def get_predictor(args):
    from neural_networks.update_manager_k import get_update_manager
    from neural_networks.recurrent_layers import get_recurrent_layers
    updater = get_update_manager(args)
    recurrent_layer = get_recurrent_layers(args)

    from neural_networks.rnn_core_keras import RNNCore

    backend = 'tensorflow'

    return RNNCore(mem_frac=args.mem_frac, backend=backend, max_length=args.max_length,
                      regularization=args.regularization, updater=updater, recurrent_layer=recurrent_layer, batch_size=args.batch_size,
                      active_f=args.act)
