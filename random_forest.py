from __future__ import print_function

import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import sys
import os

from imagery_to_data import data_gen
from unet import unet



def randomForest(data, response):
    X = data    # n x p x p (n = num samples, p = 256)            (this will eventually be convs)
    y = response    # n x 1 (n = num samples, 1 = res, 0 = sus)
    X_nu = np.zeros(shape=(X.shape[0], (X.shape[1])**2))
    print(X_nu.shape)
    for i in range(X.shape[0]):
         X_nu[i,:] = X[i,:,:].flatten()
    #print(y)
    X_train, X_test, y_train, y_test = train_test_split(X_nu, y, test_size=0.2)
    #print(y_train)
    lb = LabelBinarizer()
    y_train = np.array([number[0] for number in lb.fit_transform(y_train)])
    rf = RandomForestClassifier(n_estimators=(1000), max_features="sqrt")
    rf.fit(X_train, y_train)

    recall = cross_val_score(rf, X_test, y_test, cv=5, scoring='recall')
    precision = cross_val_score(rf, X_test, y_test, cv=5, scoring='precision')
    accuracy = cross_val_score(rf, X_test, y_test, cv=5, scoring='accuracy')
    f1_score = cross_val_score(rf, X_test, y_test, cv=5, scoring='f1_macro')

    return {'accuracy': accuracy, 'f1': f1_score, 'precision': precision, 'recall': recall}

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

def sobelFilter(X): #adds a filter from the cv2 library that makes edges easier to detect
    X_sob = np.zeros(shape=(X.shape[0], (X.shape[1]) ** 2))
    for i in range(X.shape[0]):

        img = cv2.imread(Image.fromarray(X[i,:,:]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

        edge_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        edge_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(edge_x ** 2 + edge_y ** 2)
        print(type(edge))
        X_sob[i,:] = edge
    return X_sob





p = data_gen(r"D:\2020-03-24 - ThT movies", 5, 60)

[X, y] = loadData(p)

X = np.array(X)
y = np.array(y)

X = sobelFilter(X)

print(X.shape[0])
print(X.shape[1])
print(y.shape)
dict = randomForest(X, y)

print("acc", np.mean(dict["accuracy"]))
print("f1",np.mean(dict["f1"]))
print("precision", np.mean(dict["precision"]))
print("recall", np.mean(dict["recall"]))



