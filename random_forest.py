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
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from numpy import dot
from numpy.linalg import norm


import sys, os, re, random, math

from imagery_to_data import data_gen_sobel_diff as data_gen

from unet import unet
from functions import low_activity_elim
from laplace_of_gaussian import LoGFilter
from sobel import sobelFilter

def randomForest(data, response):
    print("Begin randomForest func")
    X = data    # n x p x p (n = num samples, p = 256)
    y = response    # n x 1 (n = num samples, 1 = res, 0 = sus)
    X_nu = np.zeros(shape=(X.shape[0], (X.shape[1])**2))
    print(X_nu.shape)
    for i in range(X.shape[0]):
         X_nu[i,:] = X[i,:,:].flatten()

    X_train, X_test, y_train, y_test = train_test_split(X_nu, y, test_size=0.3)

    #lb = LabelBinarizer()
    #y_train = np.array([number[0] for number in lb.fit_transform(y_train)])
    rf = RandomForestClassifier(n_estimators=(1000), max_features="sqrt", n_jobs= -1)
    rf.fit(X_train, y_train)

    return rf, X_test, y_test


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

def trainTestByWell(dir):
    random.seed(42)
    # Resistant is class 0 and Susceptible is class 1.
    resWells = []
    susWells = []
    for well in os.listdir(dir):
        if re.search("KanR", well) or re.search("_0", well):
            print(well)
            resWells.append(os.path.join(dir,well))
            continue
        susWells.append(os.path.join(dir, well))
    random.shuffle(resWells)
    random.shuffle(susWells)

    trainsz = math.floor((len(susWells) + len(resWells)) * 0.8)
    testsz = math.ceil((len(susWells) + len(resWells)) * 0.2)

    X_train = [] #np.empty((trainsz, 256 ** 2))
    X_test = [] #np.empty((testsz, 256 **2))

    y_train = [] #np.empty(trainsz)
    y_test = [] #np.empty(testsz)

    for i in range(len(resWells)):
        if (i / len(resWells)) > 0.2:
            for pic in os.listdir(resWells[i]):
                im = Image.open(os.path.join(resWells[i], pic))
                im = ImageOps.grayscale(im)
                im = im.resize((256, 256))
                arr = np.asarray(im)

                X_nu = arr.flatten()
                X_train.append(X_nu)
                y_train.append(0)
            continue
        print(i / len(resWells))
        for pic in os.listdir(resWells[i]):
            im = Image.open(os.path.join(resWells[i], pic))
            im = ImageOps.grayscale(im)
            im = im.resize((256, 256))
            arr = np.asarray(im)


            X_nu = arr.flatten()
            X_test.append(X_nu)
            y_test.append(0)

    for j in range(len(susWells)):
        if (j / len(susWells)) > 0.2:
            for pic in os.listdir(susWells[j]):
                im = Image.open(os.path.join(susWells[j], pic))
                im = ImageOps.grayscale(im)
                im = im.resize((256, 256))
                arr = np.asarray(im)


                X_nu = arr.flatten()
                X_train.append(X_nu)
                y_train.append(1)
            continue
        for pic in os.listdir(susWells[j]):
            im = Image.open(os.path.join(susWells[j], pic))
            im = ImageOps.grayscale(im)
            im = im.resize((256, 256))
            arr = np.asarray(im)


            X_nu = arr.flatten()
            X_test.append(X_nu)
            y_test.append(1)

    train = list(zip(X_train, y_train))
    test = list(zip(X_test, y_test))

    random.shuffle(train)
    random.shuffle(test)

    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)

    return X_train, X_test, y_train, y_test





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


X_train, X_test, y_train, y_test = trainTestByWell(r'C:\Users\eugmille\Desktop\Well_test')

print(y_test)
print(y_train)
rf = RandomForestClassifier(n_estimators=(1000), max_features="sqrt", n_jobs= -1)
rf.fit(X_train, y_train)



y_pred = np.zeros(len(X_test))
y_probs = np.zeros(len(X_test))
for i in range(len(X_test)):
    y_pred[i] = rf.predict(X_test[i].reshape(1, -1))
    y_probs[i] = rf.predict_proba(X_test[i].reshape(1, -1))[:,1]


print(confusion_matrix(y_true=y_test, y_pred=y_pred))
print(roc_auc_score(y_true = y_test, y_score = y_probs))


roc_values = []
for thresh in np.linspace(0, 1, 100):
    preds = []
    for i in range(len(y_probs)):
        if y_probs[i] > thresh:
            preds.append(1)
            continue
        preds.append(0)

    # preds = map(lambda prob: prob > thresh, y_probs)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    roc_values.append([tpr, fpr])
tpr_values, fpr_values = zip(*roc_values)

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(fpr_values, tpr_values)
ax.plot(np.linspace(0, 1, 100),
         np.linspace(0, 1, 100),
         label='baseline',
         linestyle='--')
plt.title('Receiver Operating Characteristic Curve', fontsize=18)
plt.ylabel('TPR', fontsize=16)
plt.xlabel('FPR', fontsize=16)
plt.legend(fontsize=12)
plt.show()



