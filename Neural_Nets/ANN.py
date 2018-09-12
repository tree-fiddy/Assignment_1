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

from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

# Set the temporary data store for HDF
filename = '/tmp/credit_data.h5'
le = preprocessing.LabelEncoder()

csv = pd.read_csv('data.csv')
csv = csv.apply(le.fit_transform)
df = csv.to_hdf(filename, 'data', mode='w', format='table')

approved = pd.read_hdf('/tmp/credit_data.h5')
approvedX = approved.drop('A16',1).copy().values
approvedY = approved['A16'].copy().values

# madelon = pd.read_hdf('datasets.hdf','madelon')
# madelonX = madelon.drop('Class',1).copy().values
# madelonY = madelon['Class'].copy().values

approved_trgX, approved_tstX, approved_trgY, approved_tstY = ms.train_test_split(approvedX, approvedY, test_size=0.2, random_state=0,stratify=approved['A16'])

# Establish a Pipeline with repeating steps/processes
pipeA = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

d = approvedX.shape[1]
hiddens_approved = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(-1,5.01,1/2)]
params_approved = {'MLP__activation':['relu','logistic'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_approved}

# Create classifier
approved_clf = basicResults(pipeA,approved_trgX,approved_trgY,approved_tstX,approved_tstY,params_approved,'ANN','approved')

#madelon_final_params = {'MLP__hidden_layer_sizes': (500,), 'MLP__activation': 'logistic', 'MLP__alpha': 10.0}

# approved_final_params ={'MLP__hidden_layer_sizes': (30,30,30), 'MLP__activation': 'logistic', 'MLP__alpha': 0.0031622776601683794}
approved_final_params =approved_clf.best_params_
approved_OF_params =approved_final_params.copy()
approved_OF_params['MLP__alpha'] = 0.0031622776601683794
approved_OF_params['MLP__hidden_layer_sizes'] = (1)

pipeA.set_params(**approved_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})
makeTimingCurve(approvedX,approvedY,pipeA,'ANN','approved')

# pipeM.set_params(**madelon_final_params)
# pipeM.set_params(**{'MLP__early_stopping':False})
# iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','madelon')
pipeA.set_params(**approved_final_params)
pipeA.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeA,approved_trgX,approved_trgY,approved_tstX,approved_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','approved')

# pipeM.set_params(**madelon_OF_params)
# pipeM.set_params(**{'MLP__early_stopping':False})
# iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','madelon')
pipeA.set_params(**approved_OF_params)
pipeA.set_params(**{'MLP__early_stopping':False})
iterationLC(pipeA,approved_trgX,approved_trgY,approved_tstX,approved_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','approved')

