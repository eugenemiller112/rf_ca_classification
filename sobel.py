import numpy as np
import cv2

def sobelFilter(X): #adds a filter from the cv2 library that makes edges easier to detect
    X_sob = np.zeros(shape=(X.shape[0], X.shape[1], X.shape[1]))

    for i in range(X.shape[0]):

        cv2.imwrite("tmp/img.jpg", X[i, :, :])
        img = cv2.imread("tmp/img.jpg", 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)

        edge_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        edge_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.sqrt(edge_x ** 2 + edge_y ** 2)

        for j in range(edge.shape[0]):
            for k in range(edge.shape[1]):

                X_sob[i,j,k] = edge[j,k]
    return X_sob