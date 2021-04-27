'''
Hyperparameters wrapped in argparse
'''

import argparse
import os


def get_opts():
    parser = argparse.ArgumentParser(
        description='Bag of words')

    # Paths
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='data folder')
    parser.add_argument('--feat-dir', type=str, default='./feat',
                        help='feature folder')
    parser.add_argument('--out-dir', type=str, default='.',
                        help='output folder')

    # Visual words (requires tuning)
    parser.add_argument('--filter-scales', nargs='+', type=float,
                        default=[1, 2],
                        help='a list of scales for all the filters')
    parser.add_argument('--K', type=int, default=30,
                        help='# of words')
    parser.add_argument(
        '--alpha', type=int, default=32*32,
        help='Using only a subset of alpha pixels in each image'
    )

    # Recognition system (requires tuning)
    parser.add_argument('--L', type=int, default=3,
                        help='# of layers in spatial pyramid matching (SPM)')

    # Additional options (add your own hyperparameters here)

    ##
    opts = parser.parse_args()
    return opts
