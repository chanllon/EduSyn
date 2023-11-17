from pathlib import Path

import numpy
import math
import torch
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

def matrixmult(a, b):
    return numpy.dot(a, b)

def qmatrix(prolength, e):
    k = prolength
    qeye = math.exp(e)/(k-1+math.exp(e))
    qelse = 1/(k-1+math.exp(e))

    qmatrix = numpy.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                qmatrix[i][j] = qeye
            else:
                qmatrix[i][j] = qelse
    return qmatrix

def prammatrix2(k, qmatrix):

    pi = numpy.dot(numpy.linalg.inv(qmatrix), k)

    aqmatrix = numpy.zeros((len(k), len(k)))
    for i in range(len(k)):
        for j in range(len(k)):
            aqmatrix[i][j] = qmatrix[j][i]*pi[i]/k[j]
    return aqmatrix


def compund(matrixarray):
    result = matrixarray[0]
    for i in range(1, len(matrixarray)):
        result = matrixmult(result, matrixarray[i])
    return result

def train(dataset,notsecret,secret):
    data = pd.read_csv('data.csv')
    X_train, X_test, y_train, y_test = train_test_split(data[notsecret], data[secret], test_size=0.1, random_state=42)
    categorical_features_indices = numpy.where(X_train.dtypes != numpy.float)[0]
    model = CatBoostClassifier(iterations=100, depth=5,cat_features=categorical_features_indices,learning_rate=0.5, loss_function='Logloss',
                                logging_level='Verbose')
    model.fit(X_train,y_train)
    model.save_model('model.cbm')

def predict(dataset,notsecret):
    model = CatBoostClassifier()
    model.load_model('model.cbm')
    result = model.predict_proba(dataset[notsecret])
    return result


def perturb(dataset, secret, ematrix):
    perturbed = dataset
    for i in range(len(dataset)):
        for j in range(len(secret)):
            perturbed[i][secret[j]] = numpy.random.multinomial(1, ematrix[j]).argmax()
    return perturbed

def getonehot(array):
    dim = max(array)
    onehot = numpy.zeros((len(array), dim))
    for i in range(len(array)):
        onehot[i][array[i]-1] = 1
    return onehot

def tranonehot(onehot, qmatrix):
    prob = numpy.dot(qmatrix, onehot)
    for i in range(len(prob)):
        if prob[i] < 0:
            prob[i] = 0
        if prob[i] > 1:
            prob[i] = 1
    prob = prob / sum(prob)
    result = numpy.random.multinomial(1, prob).argmax()+1
    return result

def getnewarry(arry, qmatrix):
    result = []
    onehots = getonehot(arry)
    for i in range(len(onehots)):
        result.append(tranonehot(onehots[i], qmatrix))
    return result

def softmax(w):
    e = numpy.exp(numpy.array(w) / 1)
    dist = e / numpy.sum(e)
    return dist

DATA_DIR = Path.home()
s = 'ssswwe'
e = 'eeq'
print(DATA_DIR.name)
print(DATA_DIR / s / e)




