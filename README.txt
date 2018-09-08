# Assignment 1

## Decision Trees
### How to Run

-  Download dataset from: UCI Credit Approval Data Set: http://archive.ics.uci.edu/ml/datasets/credit+approval
-  Activate python 2.7 with the following packages:
    - import csv, numpy, ast
    - from custom_packages.util import entropy, information_gain, partition_classes, try_partition_classes
        -   This is a custom package I wrote for a previous course, and provided in the custom_packages
            directory

-   cd into Decision_Tree directory
-   execute the following in a terminal (assuming python2.7 environment is activated:  run_dtree.py &
-   Repeat with tweaks to the following parameters to get their respective analysis metrics:
    -   Model Complexity Analysis:  Varying forest size, all else constant
        -   At the bottom of run_dtree.py, adjust forest_size = X where X is the forest size.
    -   Learning Curves Analysis:  Varying Entropy threshold and Max Tree Depth
        -   Depth: adjust X in def split(self, node, depth, max_depth=X, min_size=5)
        -   Threshold:  adjust


## Artificial Neural Networks (ANN)
### How to Run

-   Download dataset from: UCI Credit Approval Data Set: http://archive.ics.uci.edu/ml/datasets/credit+approval
-   Required Libraries:
    -   import numpy as np
    -   from sklearn.neural_network import MLPClassifier
    -   import sklearn.model_selection as ms
    -   import pandas as pd
    -   from helper_packages.helpers import basicResults,makeTimingCurve,iterationLC
    -   from sklearn.pipeline import Pipeline
    -   from sklearn.preprocessing import StandardScaler
    -   from sklearn.ensemble import RandomForestClassifier
    -   from sklearn.feature_selection import SelectFromModel
    -   from sklearn import preprocessing