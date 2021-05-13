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

    # Visual words
    parser.add_argument('--pattern-size', type=int, default=12,
                        help='size of pattern')
    parser.add_argument('--hog-n', type=int, default=10,
                        help='n for hog')
    parser.add_argument('--alpha', type=int, default=20,
                        help='# of patterns')
    parser.add_argument('--hog-thres', type=int, default=0.18,
                        help='threshold for hog')
    parser.add_argument('--K', type=int, default=20,
                        help='# of words')

    # Recognition system
    parser.add_argument('--L', type=int, default=2,
                        help='# of layers in spatial pyramid matching (SPM)')

    # logistic regression
    parser.add_argument('--batch-size', type=int, default=2000,
                        help='batch size')
    parser.add_argument('--epoch', type=int, default=40,
                        help='# of epoches')
    parser.add_argument('--learning-rate', type=float, default=1.2e-3,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1.5e-3,
                        help='weight decay')

    # train and test
    parser.add_argument('--stack', type=int, default=3,
                        help='# of stack layers')
    parser.add_argument('--re-extract', type=int, default=1,
                        help='re-extract features')
    parser.add_argument('--feature', type=str, default='hs',
                        help='feature for classification')
    parser.add_argument('--classifier', type=str, default='lr',
                        help='classifier  for classification')
    ##
    opts = parser.parse_args()
    return opts
