from __future__ import print_function

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import sys
import os



def randomForest(data, response):
    X = data    # n x p x p (n = num samples, p = 256)
    y = response    # n x 1 (n = num samples, 1 = res, 0 = sus)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = np.array([number[0] for number in LabelBinarizer.fit_transform(y_train)])
    eval_cls = RandomForestClassifier(n_estimators=1000, max_features="sqrt")
    eval_cls.fit(X_train, y_train) 

    recall = cross_val_score(eval_cls, X_train, y_train, cv=5, scoring='recall')
    precision = cross_val_score(eval_cls, X_train, y_train, cv=5, scoring='precision')
    accuracy = cross_val_score(eval_cls, X_train, y_train, cv=5, scoring='accuracy')
    f1_score = cross_val_score(eval_cls, X_train, y_train, cv=5, scoring='f1_macro')

    return {'accuracy': accuracy, 'f1': f1_score, 'precision': precision, 'recall': recall}

def loadData(dir):
    i = 0
    X = []
    y = []
    for cat in os.listdir(dir):
        print(cat, "is class:", i)
        for datum in os.listdir(os.path.join(dir, cat)):
            im = Image.open(os.path.join(os.path.join(dir,cat),datum))
            arr = np.asarray(im)
            np.append(X, arr)
            np.append(y, i)
        i += 1
    print("Fin")
    return [X, y]

[X, y] = loadData(r'C:\Users\eugmille\Desktop\rf_test_data')

print(randomForest(X, y))



