# -*- coding: utf-8 -*-
"""
Code borrowed from Jonathan Tay
https://github.com/JonathanTay/CS-7641-assignment-1
"""
import pandas as pd
from helper_packages.helpers import  basicResults,makeTimingCurve,iterationLC,dtclf_pruned
import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
postureX = posture.drop(['Class','User'],axis=1).copy().values
postureY = posture['Class'].copy().values

# Create List of Alphas
"""
Alphas
------
Output classifier_t weight.  Is a function of the classifiers error rate.
-   This weight grows exponentially as the error approaches 0. Better classifiers are given
    exponentially more weight. 
    
Learning Rate
------
A hyperparameter (regularization parameter) that controls how much we are adjusting the weights of 
our network/model with respect to the loss gradient.
    -   The lower the alpha, the slower we travel along the downward slope.
"""

alphas =
posture_trgX, posture_tstX, posture_trgY, posture_tstY = ms.train_test_split(postureX, postureY, test_size=0.3, random_state=0,stratify=postureY)

posture_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
posture_OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)


''' Tuning '''
#paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100],
#           'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}
params_posture= {'Boost__n_estimators': [1, 2, 5, 10, 20, 30, 45, 60, 80, 100],
                 'Boost__base_estimator__alpha': [-1, -1e-1, -1e-2, -1e-3,
                                                  0,                                                  0,
                                                  1e-3, 1e-2, 1e-1, 1]
}
         
posture_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=posture_base,random_state=55)
posture_OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=posture_OF_base,random_state=55)

pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                 ('Boost',posture_booster)])

# pipeA = Pipeline([('Scale',StandardScaler()),
#                  ('Boost',adult_booster)])
#
posture_clf = basicResults(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,params_posture,'Boost','posture')
# adult_clf = basicResults(pipeA, adult_trgX, adult_trgY, adult_tstX, adult_tstY, params_posture, 'Boost', 'adult')

#
#
#posture_final_params = {'n_estimators': 20, 'learning_rate': 0.02}
#adult_final_params = {'n_estimators': 10, 'learning_rate': 1}
#OF_params = {'learning_rate':1}

posture_final_params = posture_clf.best_params_
# adult_final_params = adult_clf.best_params_
OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}

##
pipeM.set_params(**posture_final_params)
# pipeA.set_params(**adult_final_params)
makeTimingCurve(postureX,postureY,pipeM,'Boost','posture')
# makeTimingCurve(adultX,adultY,pipeA,'Boost','adult')
#
pipeM.set_params(**posture_final_params)
iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost','posture')
# pipeA.set_params(**adult_final_params)
# iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','adult')
pipeM.set_params(**OF_params)
iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost_OF','posture')