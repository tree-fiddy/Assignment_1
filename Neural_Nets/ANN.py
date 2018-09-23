"""
Code borrowed from Jonathan Tay
https://github.com/JonathanTay/CS-7641-assignment-1
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
import pandas as pd
from helper_packages.helpers import basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

# Set the temporary data store for HDF

''' Credit Card Data'''
# filename = '/tmp/credit_data.h5'
# le = preprocessing.LabelEncoder()

# csv = pd.read_csv('../data/cc_approval.csv')
# csv = csv.apply(le.fit_transform)
# df = csv.to_hdf(filename, 'data', mode='w', format='table')

# approved = pd.read_hdf('/tmp/credit_data.h5')
# approvedX = approved.drop('A16',1).copy().values
# approvedY = approved['A16'].copy().values

'''Hand Posture Data'''
csv = pd.read_csv('../data/hand_posture.csv', sep=',', header=0)
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
postureX = posture.drop(['Class','User'],axis=1).copy().values
postureY = posture['Class'].copy().values

# Train, Test, Split
# approved_trgX, approved_tstX, approved_trgY, approved_tstY = ms.train_test_split(approvedX, approvedY, test_size=0.2, random_state=0,stratify=approved['Class'])

posture_trgX, posture_tstX, posture_trgY, posture_tstY = ms.train_test_split(postureX, postureY, test_size=0.3, random_state=0,stratify=postureY)


# Establish a Pipeline with repeating steps/processes
# pipeA = Pipeline([('Scale',StandardScaler()),  
#                  ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])
pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('MLP',MLPClassifier(max_iter=2000,learning_rate='adaptive', solver='sgd',
                                      early_stopping=True,random_state=55))])

# d = approvedX.shape[1]
# hiddens_approved = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]

d = X.shape[1]
d = d//(2**4)
hiddens_posture = [10,25,50]

# params_approved = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_approved}
params_posture = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_posture}

# Create classifier
# approved_clf = basicResults(pipeA,approved_trgX,approved_trgY,approved_tstX,approved_tstY,params_approved,'ANN','approved')  # 
posture_clf = basicResults(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,params_posture,'ANN','posture')

# approved_final_params ={'MLP__hidden_layer_sizes': 50, 'MLP__activation': 'logistic', 'MLP__alpha': 0.1}
# approved_final_params =approved_clf.best_params_
# approved_OF_params =approved_final_params.copy()
# approved_OF_params['MLP__alpha'] = 0.0031622776601683794
# approved_OF_params['MLP__hidden_layer_sizes'] = (1)

# pipeA.set_params(**approved_final_params)
# pipeA.set_params(**{'MLP__early_stopping':False})
# makeTimingCurve(approvedX,approvedY,pipeA,'ANN','approved')

# posture_final_params = {'Boost__n_estimators': 50, 'Boost__learning_rate': 0.02}  
posture_final_params = posture_clf.best_params_
#OF_params = {'learning_rate':1}
posture_OF_params = {'MLP__hidden_layer_sizes':(30,30,30), 'MLP__alpha':0.00001}
makeTimingCurve(postureX,postureY,pipeM,'ANN','posture')

pipeM.set_params(**posture_final_params)
pipeM.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','posture')
# pipeA.set_params(**approved_final_params)
# pipeA.set_params(**{'MLP__early_stopping':False})
# iterationLC(pipeA,approved_trgX,approved_trgY,approved_tstX,approved_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','approved')

pipeM.set_params(**posture_OF_params)
pipeM.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','posture')
# pipeA.set_params(**approved_OF_params)
# pipeA.set_params(**{'MLP__early_stopping':False})
# iterationLC(pipeA,approved_trgX,approved_trgY,approved_tstX,approved_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','approved')

