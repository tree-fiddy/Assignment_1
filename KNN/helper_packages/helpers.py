import numpy as np
from time import clock
import sklearn.model_selection as ms
import pandas as pd
from collections import defaultdict
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier as dtclf

def balanced_accuracy(truth,pred):
    wts = compute_sample_weight('balanced',truth)
    return accuracy_score(truth,pred,sample_weight=wts)

my_scorer = make_scorer(balanced_accuracy)

def basicResults(clfObj, trgX, trgY, tstX, tstY, params, clf_type=None, dataset=None):
    np.random.seed(55)
    if clf_type is None or dataset is None:
        raise
    cv = ms.GridSearchCV(clfObj, n_jobs=10, param_grid=params, refit=True, verbose=10, cv=5, scoring=my_scorer)
    cv.fit(trgX, trgY)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./output/{}_{}_reg.csv'.format(clf_type, dataset), index=False)
    test_score = cv.score(tstX, tstY)
    with open('./output/test results.csv', 'a') as f:
        f.write('{},{},{},{}\n'.format(clf_type, dataset, test_score, cv.best_params_))
    N = trgY.shape[0]
    curve = ms.learning_curve(cv.best_estimator_, trgX, trgY, cv=5,
                              train_sizes=[500, 100] + [int(N * x / 10) for x in range(1, 8)], verbose=10,
                              scoring=my_scorer)
    curve_train_scores = pd.DataFrame(index=curve[0], data=curve[1])
    curve_test_scores = pd.DataFrame(index=curve[0], data=curve[2])
    curve_train_scores.to_csv('./output/{}_{}_LC_train.csv'.format(clf_type, dataset))
    curve_test_scores.to_csv('./output/{}_{}_LC_test.csv'.format(clf_type, dataset))
    return cv


def makeTimingCurve(X, Y, clf, clfName, dataset):
    out = defaultdict(dict)
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=frac, random_state=42)
        st = clock()
        np.random.seed(55)
        clf.fit(X_train, y_train)
        out['train'][frac] = clock() - st
        st = clock()
        clf.predict(X_test)
        out['test'][frac] = clock() - st
        print(clfName, dataset, frac)
    out = pd.DataFrame(out)
    out.to_csv('./output/{}_{}_timing.csv'.format(clfName, dataset))
    return