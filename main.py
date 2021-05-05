from os.path import join
from time import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import util
import visual_recog
import visual_words
from opts import get_opts
import scipy.io


def main():

    opts = get_opts()

    # opts.filter_scales = [1, 2, 5]

    # Q1.1
    # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
    # img_path = join(opts.data_dir, 'aquarium/sun_aztvjgubyrgvirup.jpg')
    # img = Image.open(img_path)
    # img = np.array(img).astype(np.float32) / 255

    
    train_data = scipy.io.loadmat('./data/nist36_train.mat')
    train_x, train_y = train_data['train_data'], train_data['train_labels']
    img = train_x[1, ...].reshape((32, 32))
    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses)

    '''for grid search'''
    for filter_scale in list([[1, 2], [1, 2, 5]]):
        print(filter_scale)
        opts.filter_scales = filter_scale
        for K in range(10, 211, 50):
            opts.K = K

            # indent line 40-89 twice after uncomment line 36-38
            # for alpha in range(25, 51, 25):
            #     opts.alpha = alpha
                # for i in range(1):

            start_tm = time()


            # Q1.2
            n_cpu = util.get_num_CPU()
            visual_words.compute_dictionary(opts, n_worker=n_cpu)

            # Q1.3
            # img_path = join(opts.data_dir, 'kitchen/sun_aasmevtpkslccptd.jpg')
            # img_path = join(opts.data_dir, 'aquarium/sun_aairflxfskjrkepm.jpg')
            # img_path = join(opts.data_dir, 'windmill/sun_atkgjktpdtiaanmm.jpg')
            # img_path = join(opts.data_dir, 'desert/sunbmutmkauvllgeuwj.jpg')
            # img = Image.open(img_path)
            # img = np.array(img).astype(np.float32)/255
            # dictionary = np.load(join(opts.out_dir, 'dictionary.npy'))
            # wordmap = visual_words.get_visual_words(opts, img, dictionary)
            # util.visualize_wordmap(img, wordmap)

            for L in range(1, 3):
                opts.L = L

                # Q2.1-2.4
                n_cpu = util.get_num_CPU()
                visual_recog.build_recognition_system(
                    opts, n_worker=n_cpu)

                # Q2.5
                n_cpu = util.get_num_CPU()
                conf, accuracy = visual_recog.evaluate_recognition_system(
                    opts, n_worker=n_cpu)

                print(conf)
                print(accuracy)

                # # show time used
                # second_used = time() - start_tm
                # print('Time used = {:.0f} min {:.0f} s'.format(
                #     second_used//60, second_used % 60))

                # # save parameters
                # f = open(join(opts.out_dir, 'myresult.csv'), 'a')
                # f.write(str(opts.filter_scales).replace(',', ';')+',{},{},{},
                #   {},{:.0f}\'{:.0f}\'\'\n'.format(opts.K, opts.alpha, opts.L, 
                #   accuracy, second_used//60, second_used % 60))
                # f.close()


if __name__ == '__main__':
    main()
