# -*- coding: utf-8 -*-
"""
Code borrowed from Jonathan Tay
https://github.com/JonathanTay/CS-7641-assignment-1
"""
import sklearn.model_selection as ms
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from helper_packages.helpers import  basicResults,makeTimingCurve,iterationLC,dtclf_pruned
from sklearn.ensemble import RandomForestClassifier
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
postureX = posture.drop('Class',axis=1).copy().values
postureY = posture['Class'].copy().values

# Create List of Alphas
alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]

posture_trgX, posture_tstX, posture_trgY, posture_tstY = ms.train_test_split(postureX, postureY, test_size=0.3, random_state=0,stratify=postureY)

adult_base = dtclf_pruned(criterion='entropy',class_weight='balanced',random_state=55)
OF_base = dtclf_pruned(criterion='gini',class_weight='balanced',random_state=55)                
#paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,40,50],'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}
paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
          'Boost__base_estimator__alpha':alphas}
#paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100],
#           'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}

paramsM = {'Boost__n_estimators':[1,2,5,10,20,30,45,60,80,100],
           'Boost__base_estimator__alpha':alphas}
                                   
         
madelon_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=madelon_base,random_state=55)
adult_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=adult_base,random_state=55)
OF_booster = AdaBoostClassifier(algorithm='SAMME',learning_rate=1,base_estimator=OF_base,random_state=55)

pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                 ('Boost',madelon_booster)])

pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('Boost',adult_booster)])

#
madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,paramsM,'Boost','madelon')        
adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,paramsA,'Boost','adult')        

#
#
#madelon_final_params = {'n_estimators': 20, 'learning_rate': 0.02}
#adult_final_params = {'n_estimators': 10, 'learning_rate': 1}
#OF_params = {'learning_rate':1}

madelon_final_params = madelon_clf.best_params_
adult_final_params = adult_clf.best_params_
OF_params = {'Boost__base_estimator__alpha':-1, 'Boost__n_estimators':50}

##
pipeM.set_params(**madelon_final_params)
pipeA.set_params(**adult_final_params)
makeTimingCurve(madelonX,madelonY,pipeM,'Boost','madelon')
makeTimingCurve(adultX,adultY,pipeA,'Boost','adult')
#
pipeM.set_params(**madelon_final_params)
iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost','madelon')        
pipeA.set_params(**adult_final_params)
iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost','adult')                
pipeM.set_params(**OF_params)
iterationLC(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50,60,70,80,90,100]},'Boost_OF','madelon')                
pipeA.set_params(**OF_params)
iterationLC(pipeA,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,{'Boost__n_estimators':[1,2,5,10,20,30,40,50]},'Boost_OF','adult')                

             