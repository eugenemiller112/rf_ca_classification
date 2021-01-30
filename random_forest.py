from __future__ import print_function

import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import scipy.misc as sc
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import sys
import os

from imagery_to_data import data_gen_diff as data_gen
from imagery_to_data import data_gen_sobel_diff as data_gen2
from imagery_to_data import data_gen_LoG_diff as data_gen3
from unet import unet
from functions import low_activity_elim
from laplace_of_gaussian import LoGFilter
from sobel import sobelFilter

def randomForest(data, response):
    print("Begin randomForest fun")
    X = data    # n x p x p (n = num samples, p = 256)
    y = response    # n x 1 (n = num samples, 1 = res, 0 = sus)
    X_nu = np.zeros(shape=(X.shape[0], (X.shape[1])**2))
    print(X_nu.shape)
    for i in range(X.shape[0]):
         X_nu[i,:] = X[i,:,:].flatten()

    X_train, X_test, y_train, y_test = train_test_split(X_nu, y, test_size=0.2)

    #lb = LabelBinarizer()
    #y_train = np.array([number[0] for number in lb.fit_transform(y_train)])
    rf = RandomForestClassifier(n_estimators=(1000), max_features="sqrt")
    rf.fit(X_train, y_train)

    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    recall = cross_val_score(rf, X_test, y_test, cv=5, scoring='recall')
    precision = cross_val_score(rf, X_test, y_test, cv=5, scoring='precision')
    accuracy = cross_val_score(rf, X_test, y_test, cv=5, scoring='accuracy')
    f1_score = cross_val_score(rf, X_test, y_test, cv=5, scoring='f1_macro')

    return [{'accuracy': accuracy, 'f1': f1_score, 'precision': precision, 'recall': recall}, rf]

def loadData(dir):
    i = 0
    X = []
    y = []
    for cat in os.listdir(dir):
        print(cat, "is class:", i)
        for datum in os.listdir(os.path.join(dir, cat)):
            im = Image.open(os.path.join(os.path.join(dir,cat),datum))
            im = ImageOps.grayscale(im)
            im = im.resize((256,256))
            arr = np.asarray(im)
            X.append(arr)
            y.append(i)
        i += 1
    print("Data Loaded!")
    return [X, y]

def generateCNN(X, pretrained_weights = None): #generates convolutional layers based off the data
    if pretrained_weights is None:
        print("Warning, no model has been loaded")
    mod = unet(pretrained_weights)  # load model

    # prepare the data
    X_nu = np.zeros(shape=(X.shape[0], (X.shape[1])**2))
    print(X_nu.shape)
    for i in range(X.shape[0]):
         X_nu[i,:] = mod.predict(X[i,:,:])
    return X_nu


res = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'deltaT': []}
# p = data_gen(r"D:\RFtrain", 5, 180)
p = r"D:\ASD\01-22-2021, 15-29-12"

[X, y] = loadData(p)

X = np.array(X)
y = np.array(y)

X_1 = sobelFilter(X)
X_2 = LoGFilter(X)


dict_1, rf_1 = randomForest(X_1, y)
dict_2, rf_2 = randomForest(X_2, y)

print("acc", np.mean(dict_1["accuracy"]))
print("f1",np.mean(dict_1["f1"]))
print("precision", np.mean(dict_1["precision"]))
print("recall", np.mean(dict_1["recall"]))

print("acc 2", np.mean(dict_2["accuracy"]))
print("f1 2",np.mean(dict_2["f1"]))
print("precision 2", np.mean(dict_2["precision"]))
print("recall 2", np.mean(dict_2["recall"]))

