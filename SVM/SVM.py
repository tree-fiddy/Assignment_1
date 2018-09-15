
# -*- coding: utf-8 -*-
"""
Code borrowed from Jonathan Tay
https://github.com/JonathanTay/CS-7641-assignment-1
"""

import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from helper_packages.helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

class primalSVM_RBF(BaseEstimator, ClassifierMixin):
    """http://scikit-learn.org/stable/developers/contributing.html"""
    
    def __init__(self, alpha=1e-9,gamma_frac=0.1,n_iter=2000):
         self.alpha = alpha
         self.gamma_frac = gamma_frac
         self.n_iter = n_iter
         
    def fit(self, X, y):
        """Check that X and y have correct shape"""
        X, y = check_X_y(X, y)

        # Get the kernel matrix
        dist = euclidean_distances(X, squared=True)
        median = np.median(dist)
        del dist
        gamma = median
        gamma *= self.gamma_frac
        self.gamma = 1 / gamma
        kernels = rbf_kernel(X, None, self.gamma)

        self.X_ = X
        self.classes_ = unique_labels(y)
        self.kernels_ = kernels
        self.y_ = y
        self.clf = SGDClassifier(loss='hinge', penalty='l2', alpha=self.alpha,
                                 l1_ratio=0, fit_intercept=True, verbose=False,
                                 average=False, learning_rate='optimal',
                                 class_weight='balanced', n_iter=self.n_iter,
                                 random_state=55)
        self.clf.fit(self.kernels_, self.y_)

        # Return the classifier
        return self


    def predict(self, X):
         # Check is fit had been called
         check_is_fitted(self, ['X_', 'y_','clf','kernels_'])
         # Input validation
         X = check_array(X)
         new_kernels = rbf_kernel(X,self.X_,self.gamma )
         pred = self.clf.predict(new_kernels)
         return pred


'''Hand Gesture Data'''
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

posture_trgX, posture_tstX, posture_trgY, posture_tstY = ms.train_test_split(postureX, postureY, test_size=0.3, random_state=0,stratify=postureY)

N_posture = posture_trgX.shape[0]

alphas = [10**-x for x in np.arange(1,9.01,1/2)]


# Linear SVM (Parametric)
pipeM = Pipeline([('Scale',StandardScaler()),
                ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])

params_posture = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_posture)/.8)+1]}

posture_clf = basicResults(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,params_posture,'SVM_Lin','posture')

#posture_final_params = {'SVM__alpha': 0.031622776601683791, 'SVM__n_iter': 687.25}
posture_final_params = posture_clf.best_params_
posture_OF_params = {'SVM__n_iter': 1000, 'SVM__alpha': 1e-16}

pipeM.set_params(**posture_final_params)
makeTimingCurve(postureX,postureY,pipeM,'SVM_Lin','posture')

pipeM.set_params(**posture_final_params)
iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'SVM__n_iter':[2**x for x in range(12)]},'SVM_Lin','posture')

pipeM.set_params(**posture_OF_params)
iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'SVM__n_iter':np.arange(100,2600,100)},'SVM_LinOF','posture')





#
# #RBF SVM (More Complex.  Non-Parametric)
# gamma_fracsA = np.arange(0.2,2.1,0.2)
# gamma_fracsM = np.arange(0.05,1.01,0.1)
#
# #
# pipeM = Pipeline([('Scale',StandardScaler()),
#                  ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
#                  ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
#                  ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
#                  ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
#                  ('SVM',primalSVM_RBF())])
#
# pipeA = Pipeline([('Scale',StandardScaler()),
#                  ('SVM',primalSVM_RBF())])
#
#
# params_adult = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_adult)/.8)+1],'SVM__gamma_frac':gamma_fracsA}
# params_posture = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_posture)/.8)+1],'SVM__gamma_frac':gamma_fracsM}
# #
# posture_clf = basicResults(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,params_posture,'SVM_RBF','posture')
# adult_clf = basicResults(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,params_adult,'SVM_RBF','adult')
#
#
#
# posture_final_params = posture_clf.best_params_
# posture_OF_params = posture_final_params.copy()
# posture_OF_params['SVM__alpha'] = 1e-16
# adult_final_params =adult_clf.best_params_
# adult_OF_params = adult_final_params.copy()
# adult_OF_params['SVM__alpha'] = 1e-16
#
# pipeM.set_params(**posture_final_params)
# makeTimingCurve(postureX,postureY,pipeM,'SVM_RBF','posture')
# pipeA.set_params(**adult_final_params)
# makeTimingCurve(adultX,adultY,pipeM,'SVM_RBF','adult')
#
#
# pipeM.set_params(**posture_final_params)
# iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'SVM__n_iter':[2**x for x in range(12)]},'SVM_RBF','posture')
# pipeA.set_params(**adult_final_params)
# iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF','adult')
#
# pipeA.set_params(**adult_OF_params)
# iterationLC(pipeA,adult_trgX,adult_trgY,adult_tstX,adult_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','adult')
# pipeM.set_params(**posture_OF_params)
# iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'SVM__n_iter':np.arange(100,2600,100)},'SVM_RBF_OF','posture')
