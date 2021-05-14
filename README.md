# 18752s21-handwritten-char

## Preprocessing:
Preprocess raw images into black and white images of single character or digit.
### 
    preprocessing.py

## Features:
Extract HOG-stack features
### 
    extract_hog_stack.py
Extract HOG-corner and BOW features
### 
    extract_hog_n_bow.py
Extract encoded features
###
    extract_autoencoder.py
Extract zernike features
###
    extract_zernike.py

## Bag of visual words:
Helper functions for *extract_hog_stack.py*
### 
    helper.py

## Visualization:
Visualize features by SVD
### 
    visualize_by_svd.py
Visualize wordmap
### 
    visualize_bow.py

## Classification:
Train and test specific features with specific classifier
### 
    train_n_test.py
Abalation study for HOG-corner parameters (Validation)
### 
    train_n_test_hog_corner.py

## CNN:
### 
    CNN/config.py
    CNN/data.py
    CNN/network.py
    CNN/train.py
    CNN/test.py

## Other:
Arguments:
### 
    opts.py
###
    # Paths
    data folder: --data-dir, type=str, default='./data/classified'
                 --data-dir s # to use sample data
    feature folder: --feat-dir, type=str, default='./feat'
    output folder: --out-dir, type=str, default='./output'

    # Visual words
    size of pattern: --pattern-size, type=int, default=12
    n for skimage.feature.hog--hog-n, type=int, default=10
    threshold for skimage.feature.hog: --hog-thres, type=int, default=0.18
    # of patterns: --alpha, type=int, default=20
    # of words: --K, type=int, default=20

    # Recognition system
    # of layers in spatial pyramid matching (SPM): --L, type=int, default=2

    # logistic regression
    batch size: --batch-size, type=int, default=2000
    # of epoches: --epoch, type=int, default=40
    learning rate: --learning-rate, type=float, default=1.2e-3
    weight decay: --weight-decay, type=float, default=1.5e-3

    # train and test
    # of stack layers: --stack, type=int, default=3
    re-extract features: --re-extract, type=int, default=1
                         --re-extract 0 # to use extracted features
    feature choosed: --feature, type=str, default='hs'
                     --feature hs # HOG-stack
                     --feature hc # HOG-corner
                     --feature bow # BOW
                     --feature orig # flattend image pixels
                     --feature ae # Autoencoder
                     --feature z # zernike
    classifier  for classification: --classifier, type=str, default='lr'
                                    --classifier lr # logistic regression
                                    --classifier lda # LDA
                                    --classifier qda # QDA
                                    --classifier gnb # Gaussian Naive Bayes
Helper functions for loading and ploting data and features
### 
    util.py
