import pandas as pd
import numpy
import math
import torch
from catboost import CatBoostClassifier
from catboost import MultiTargetCustomObjective
import pandas as pd
from sklearn.model_selection import train_test_split
import method


data = pd.read_csv('data/ukm/ukm.csv')

# secret = 'Target'
secret = 'UNS'

notsecret = data.columns.drop(secret)

qnum0 = 4
e1 = 0.1

qm = method.qmatrix(qnum0,e1)

secret_data = data[secret].astype(int)
newsecret = method.getnewarry(secret_data,qm)

data[secret] = newsecret

sec_opt= 4
records = len(data)

X_train1, X_test1, y_train1, y_test1 = train_test_split(data[notsecret], data[secret], test_size=0.1, random_state=42)


categorical_features_indices = numpy.where(X_train1.dtypes != float)[0]
model1 = CatBoostClassifier(iterations=500, depth=5,cat_features=categorical_features_indices,learning_rate=0.02, loss_function='MultiClass',logging_level='Verbose')
model1.fit(X_train1,y_train1)
model1.save_model('data/ukm/model1.cbm')
result1 = model1.predict_proba(data[notsecret])


model1 = CatBoostClassifier()
model1.load_model('data/ukm/model1.cbm')


result1 = model1.predict_proba(data[notsecret])
reverse1 = numpy.dot(result1,numpy.linalg.inv(qm))

secondpm1 = []

for k in range(len(reverse1)):
    k1 = result1[k]
    pi1 = reverse1[k]
    aqmatrix1 = numpy.zeros((sec_opt, sec_opt))
    for i in range(sec_opt):
        for j in range(sec_opt):
            aqmatrix1[i][j] = qm[j][i] * pi1[i] / k1[j]
    secondpm1.append(aqmatrix1)

secret_data = data[secret]
newsecret = []
onehots = method.getonehot(secret_data)
for j in range(records):
    newsecret.append(method.tranonehot(onehots[j], secondpm1[j]))
data[secret] = newsecret


data.to_csv('data/ukm/tradeoff/ukmpram_e001.csv',index=False)
