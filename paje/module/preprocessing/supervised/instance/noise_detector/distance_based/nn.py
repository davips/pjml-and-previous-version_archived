import math
import numpy as np

from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

# TODO: port this to Pajé
def ENN(X, y, k):

    neigh = KNeighborsClassifier(n_neighbors=k, weights='uniform', 
        algorithm='brute')
    neigh.fit(X, y)
    pred = neigh.predict(X)

    noise = []
    for i in range(len(X)):
        if pred[i] != y[i]:
            noise.append(i)

    X = np.delete(X, noise, axis=0)
    y = np.delete(y, noise)

    return X, y

def RENN(X, y, k):

    noise = []

    while(True):

        X = np.delete(X, noise, axis=0)
        y = np.delete(y, noise)

        neigh = KNeighborsClassifier(n_neighbors=k, weights='uniform', 
            algorithm='brute')
        neigh.fit(X, y)
        pred = neigh.predict(X)

        noise = []
        for i in range(len(X)):
            if pred[i] != y[i]:
                noise.append(i)

        if len(noise) == 0 or len(X) == 0:
            break

    return X, y

def consensus(pred, y):

    noise = []
    for i in range(len(X)):

        aux = Counter(pred[:,i])
        tmp = [i for i, e in enumerate(aux.values()) if e == len(pred)]

        if len(tmp) != 0:
            if list(aux.keys())[tmp[0]] != y[i]:
                noise.append(i)

    return noise

def majority(pred, y):

    noise = []
    for i in list(range(len(X))):

        aux = Counter(pred[:,i])
        tmp = [i for i, e in enumerate(aux.values()) if e > len(pred)/2]

        if len(tmp) != 0:
            if list(aux.keys())[tmp[0]] != y[i]:
                noise.append(i)

    return noise

def AENN(X, y, k, vote='majority'):

    votes = []
    for i in range(1, k+1):

        neigh = KNeighborsClassifier(n_neighbors=i, weights='uniform', 
            algorithm='brute')
        neigh.fit(X, y)
        pred = neigh.predict(X)
        votes.append(pred)

    votes = np.asarray(votes)

    noise = locals()[vote](votes, y)
    X = np.delete(X, noise, axis=0)
    y = np.delete(y, noise)

    return X, y
