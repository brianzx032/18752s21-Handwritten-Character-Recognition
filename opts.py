'''
Hyperparameters wrapped in argparse
Reference: 16720 Computer Vision S21 HW1
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
    parser.add_argument('--out-dir', type=str, default='./output',
                        help='output folder')

    # Visual words (requires tuning)
    parser.add_argument('--pattern-size', type=int, default=7,
                        help='size of pattern')
    parser.add_argument('--alpha', type=int, default=20,
                        help='# of patterns')
    parser.add_argument('--hog-thres', type=int, default=0.15,
                        help='threshold for hog')
    parser.add_argument('--K', type=int, default=10,
                        help='# of words')

    # Recognition system (requires tuning)
    parser.add_argument('--L', type=int, default=3,
                        help='# of layers in spatial pyramid matching (SPM)')

    # logistic regression
    parser.add_argument('--batch-size', type=int, default=1500,
                        help='batch size')
    parser.add_argument('--epoch', type=int, default=20,
                        help='# of epoches')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='weight decay')

    ##
    opts = parser.parse_args()
    return opts
