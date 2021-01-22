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

from imagery_to_data import data_gen
from unet import unet

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

def sobelFilter(X): #adds a filter from the cv2 library that makes edges easier to detect
    X_sob = np.zeros(shape=(X.shape[0], X.shape[1], X.shape[1]))

    for i in range(X.shape[0]):

        print(X.shape)
        print(X_sob.shape)


        cv2.imwrite("tmp/img.jpg", X[i, :, :])
        img = cv2.imread("tmp/img.jpg", 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

        edge_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        edge_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(edge_x ** 2 + edge_y ** 2)
        print(edge.shape)
        print(edge)

        for j in range(edge.shape[0]):
            for k in range(edge.shape[1]):
                #print("Shapes")
                #print(X_sob.shape)
                #print(edge.shape)
                X_sob[i,j,k] = edge[j,k]
    return X_sob

res = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'deltaT': []}

p = data_gen(r"D:\RFtrain", 5, 180)

[X, y] = loadData(p)

X = np.array(X)
y = np.array(y)

X = sobelFilter(X)

print(X.shape[0])
print(X.shape[1])
print(y.shape)
dict, rf = randomForest(X, y)

print("acc", np.mean(dict["accuracy"]))
print("f1",np.mean(dict["f1"]))
print("precision", np.mean(dict["precision"]))
print("recall", np.mean(dict["recall"]))

#for i in range(30,180,10):
#    p = data_gen(r"D:\RFtrain", 5, i)
#
#    [X, y] = loadData(p)
#
#    X = np.array(X)
#    y = np.array(y)
#
#    X = sobelFilter(X)
#    X_nu = np.zeros(shape=(X.shape[0], (X.shape[1]) ** 2))
#    for i in range(X.shape[0]):
#         X_nu[i,:] = X[i,:,:].flatten()
#
#    recall = cross_val_score(rf, X_nu, y, cv=5, scoring='recall')
#    precision = cross_val_score(rf, X_nu, y, cv=5, scoring='precision')
#    accuracy = cross_val_score(rf, X_nu, y, cv=5, scoring='accuracy')
#    f1_score = cross_val_score(rf, X_nu, y, cv=5, scoring='f1_macro')
#
#    res["accuracy"].append(np.mean(accuracy))
#    res["f1"].append(np.mean(f1_score))
#    res["precision"].append(np.mean(precision))
#    res["recall"].append(np.mean(recall))
#    res["deltaT"].append(i)
#
#
#plt.figure(figsize=(10, 10))
#plt.plot(res['deltaT'], res['accuracy'])
#plt.xlabel("delta T")
#plt.ylabel("Accuracy")
#plt.show()
#
#plt.figure(figsize=(10, 10))
#plt.plot(res['deltaT'], res['f1'])
#plt.xlabel("delta T")
#plt.ylabel("F1 Score")
#plt.show()
#
#plt.figure(figsize=(10, 10))
#plt.plot(res['deltaT'], res['precision'])
#plt.xlabel("delta T")
#plt.ylabel("Precision")
#plt.show()
#
#plt.figure(figsize=(10, 10))
#plt.plot(res['deltaT'], res['recall'])
#plt.xlabel("delta T")
#plt.ylabel("Recall")
#plt.show()