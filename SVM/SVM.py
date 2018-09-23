
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
from sklearn import preprocessing

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

''' Credit Card Data'''
filename = '/tmp/credit_data.h5'
le = preprocessing.LabelEncoder()

csv = pd.read_csv('../data/cc_approval.csv')
csv = csv.apply(le.fit_transform)
df = csv.to_hdf(filename, 'data', mode='w', format='table')

approved = pd.read_hdf('/tmp/credit_data.h5')
approvedX = approved.drop('A16',1).copy().values
approvedY = approved['A16'].copy().values

'''Hand Gesture Data'''
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
postureX = posture.drop(['Class'],axis=1).copy().values
postureY = posture['Class'].copy().values

approved_trgX, approved_tstX, approved_trgY, approved_tstY = ms.train_test_split(approvedX, approvedY, test_size=0.2, random_state=0,stratify=approved['A16'])
N_approved = approved_trgX.shape[0]
posture_trgX, posture_tstX, posture_trgY, posture_tstY = ms.train_test_split(postureX, postureY, test_size=0.3, random_state=0,stratify=postureY)
N_posture = posture_trgX.shape[0]

# #
# # """ Linear SVM (Parametric) """
# pipeA = Pipeline([('Scale',StandardScaler()),
#                   ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])

# # pipeM = Pipeline([('Scale',StandardScaler()),
# #                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
# #                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
# #                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
# #                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
# #                 ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])
#
# params_approved = {'SVM__alpha':[100, 10, 1, 0.1, 0.001, 0.0001]}
# # params_posture = {'SVM__alpha':[100, 10, 1, 0.1, 0.001, 0.0001]}
#
# approved_clf = basicResults(pipeA,approved_trgX,approved_trgY,approved_tstX,approved_tstY,params_approved,'SVM_Lin','approved')
# # posture_clf = basicResults(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,params_posture,'SVM_Lin','posture')
#
# posture_final_params = {'SVM__alpha': 0.0001}
# # posture_final_params = posture_clf.best_params_
# posture_OF_params = {'SVM__alpha': 1e-16}
#
# pipeM.set_params(**posture_final_params)
# makeTimingCurve(postureX,postureY,pipeM,'SVM_Lin','posture')
#
# pipeM.set_params(**posture_final_params)
# iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'SVM__n_iter':[2**x for x in range(12)]},'SVM_Lin','posture')
#
# pipeM.set_params(**posture_OF_params)
# iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'SVM__n_iter':np.arange(100,2600,100)},'SVM_LinOF','posture')

"""" Adding Complexity via RBF Kernel (Non-Parametric)"""
gamma_fracsM = np.arange(0.1,1,10)

pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                    ('SVM',primalSVM_RBF())])

pipeA = Pipeline([('Scale',StandardScaler()),
                 ('SVM',primalSVM_RBF())])

params_approved = {'SVM__alpha':[0.1, 0.01, 0.001],'SVM__gamma_frac':gamma_fracsM}
params_posture = {'SVM__alpha':[0.1, 0.01, 0.001],'SVM__gamma_frac':gamma_fracsM}
approved_clf = basicResults(pipeA,approved_trgX,approved_trgY,approved_tstX,approved_tstY,params_approved,'SVM_RBF','approved')
posture_clf = basicResults(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,params_posture,'SVM_RBF','posture')

posture_final_params = posture_clf.best_params_
posture_OF_params = posture_final_params.copy()
posture_OF_params['SVM__alpha'] = 1e-16
approved_final_params =approved_clf.best_params_
approved_OF_params = approved_final_params.copy()a
approved_OF_params['SVM__alpha'] = 1e-16

pipeM.set_params(**posture_final_params)
makeTimingCurve(postureX,postureY,pipeM,'SVM_RBF','posture')
pipeA.set_params(**approved_final_params)
makeTimingCurve(approvedX,approvedY,pipeM,'SVM_RBF','approved')


pipeM.set_params(**posture_final_params)
iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'SVM__n_iter':[2**x for x in range(12)]},'SVM_RBF','posture')
pipeA.set_params(**approved_final_params)
iterationLC(pipeA,approved_trgX,approved_trgY,approved_tstX,approved_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF','approved')

pipeA.set_params(**approved_OF_params)
iterationLC(pipeA,approved_trgX,approved_trgY,approved_tstX,approved_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','approved')
pipeM.set_params(**posture_OF_params)
iterationLC(pipeM,posture_trgX,posture_trgY,posture_tstX,posture_tstY,{'SVM__n_iter':np.arange(100,2600,100)},'SVM_RBF_OF','posture')
