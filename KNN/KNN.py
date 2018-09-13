# -*- coding: utf-8 -*-
"""
Code borrowed from Jonathan Tay
https://github.com/JonathanTay/CS-7641-assignment-1

Performs k-NN on Hand Gesture Posture Data

"""

import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helper_packages.helpers import basicResults,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from sklearn import preprocessing


## Hand Gesture Data
csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00391/allUsers.lcl.csv')
csv.replace('?', 0, inplace=True)

# Only the first 5 vars get converted to float (others being kept as 'object')
# This fixes that problem
csv = csv.astype(float)
csv['Class'] = csv['Class'].astype(int)

# Convert to HDF5 for high efficiency memory usage
filename = '/tmp/posture_data.h5'
df = csv.to_hdf(filename, 'data', mode='w', format='table')
posture = pd.read_hdf('/tmp/posture_data.h5')
posture = posture.iloc[1:]
postureX = posture.drop('Class',axis=1).copy().values
postureY = posture['Class'].copy().values

posture_trgX, posture_tstX, posture_trgY, posture_tstY = ms.train_test_split(postureX, postureY, test_size=0.3, random_state=0,stratify=postureY)
pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('KNN',knnC())])

# params_posture= {'KNN__metric':['manhattan','euclidean'],'KNN__n_neighbors':np.arange(1,51,5),'KNN__weights':['uniform','distance']}
params_posture= {'KNN__metric':['manhattan','euclidean'],'KNN__n_neighbors':[np.arange(1,51,5)],'KNN__weights':['uniform','distance']}


posture_clf = basicResults(pipeM,
                           posture_trgX,
                           posture_trgY,
                           posture_tstX,
                           posture_tstY,
                           params_posture,'KNN','posture')

# posture_final_params={'KNN__n_neighbors': 500, 'KNN__weights': 'distance', 'KNN__p': 1}
posture_final_params=posture_clf.best_params_
pipeM.set_params(**posture_final_params)
makeTimingCurve(postureX,postureY,pipeM,'KNN','posture')

# adult = pd.read_hdf('datasets.hdf','adult')
# adultX = adult.drop('income',1).copy().values
# adultY = adult['income'].copy().values

# adult_trgX, adult_tstX, adult_trgY, adult_tstY = ms.train_test_split(adultX, adultY, test_size=0.3, random_state=0,stratify=adultY)
# d = adultX.shape[1]

# pipeA = Pipeline([('Scale',StandardScaler()),
#                  ('KNN',knnC())])
# params_adult= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
# adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'KNN','adult')


#adult_final_params={'KNN__n_neighbors': 142, 'KNN__p': 1, 'KNN__weights': 'uniform'}
# adult_final_params=adult_clf.best_params_
# pipeA.set_params(**adult_final_params)
# makeTimingCurve(adultX,adultY,pipeA,'KNN','adult')