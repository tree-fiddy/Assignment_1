# Assignment 1
Note:  The majority of the codes used were borrowed from either previous coursework
or from https://github.com/JonathanTay

It is important to place data in the data directory, as all code references this location
to pull data.


Note:
## Decision Trees
### How to Run

-  Download dataset from: UCI Credit Approval Data Set: http://archive.ics.uci.edu/ml/datasets/credit+approval
-  Activate python 2.7 with the following packages:
    - import csv, numpy, ast
    - from custom_packages.util import entropy, information_gain, partition_classes, try_partition_classes
        -   This is a custom package I wrote for a previous course, and provided in the custom_packages
            directory

-   cd into Decision_Tree directory
-   execute the following in a terminal (assuming python2.7 environment is activated:  random_forest.py &
    -   This executes using the Credit Approval Data
    -   Run again using hand_random_forest.py &
-   Repeat with tweaks to the following parameters to get their respective analysis metrics:
    -   Model Complexity Analysis:  Varying forest size, all else constant
        -   At the bottom of run_dtree.py, adjust forest_size = X where X is the forest size.
    -   Learning Curves Analysis:  Varying Entropy threshold and Max Tree Depth
        -   Depth: adjust X in def split(self, node, depth, max_depth=X, min_size=5)
        -   Threshold:  adjust


## Artificial Neural Networks (ANN)
### How to Run

-   Download dataset from: UCI Credit Approval Data Set: http://archive.ics.uci.edu/ml/datasets/credit+approval
-   Activate Python 3.6 with Required Libraries:
    -   import numpy as np
    -   from sklearn.neural_network import MLPClassifier
    -   import sklearn.model_selection as ms
    -   import pandas as pd
    -   from helper_packages.helpers import basicResults,makeTimingCurve,iterationLC
        -   ** Custom Package downloaded from JonathanTay Github
    -   from sklearn.pipeline import Pipeline
    -   from sklearn.preprocessing import StandardScaler
    -   from sklearn.ensemble import RandomForestClassifier
    -   from sklearn.feature_selection import SelectFromModel
    -   from sklearn import preprocessing

-   Execute the following command from the ANN directory:
    -   python ANN.py &
    -   Many .csv outputs will be provided that will be necessary when comparing
        performance across different parameters
    - Parameter Tuning:  By default, we specify .best_params_ for GridSearch to discover the best
        parameters.
            -   Learning Curve: To analyze the different parameters and hyperparameters, make sure to hard code the
                parameters/hyperparameters inside params_posture/params_approved based on what values
                you'd like GridSearch to iterate.

            -   Model Complexity: Where applicable, one can use the parameters uncovered by .best_params_ and hard-code
                the attributes to the <dataset>_final_params & <dataset>_OF_params variables to run:
                    -   makeTimingCurve
                    -   iterationLC
                OF's main goal is to purposely OverFit the data, so choose hyperparameters that wouldn't surprise you if it
                overfits.
-   Graphing:
    -   Many .csv files will be spit out.  Use the <model>_<dataset>_LC_train & <model>_<dataset>_LC_test csv files to
        plot in Excel to plot the learning curves and model complexity curves.
        -   Where applicable, use the ITER csvs to plot the Hyperparameters, where iterations are telling of a story.

## KNN
### How to Run

-   Data will download automatically by the following command:
    -   csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00391/allUsers.lcl.csv')
-   Activate Python 3.6 with Required Libraries:
    -   import numpy as np
    -   import sklearn.model_selection as ms
    -   from sklearn.neighbors import KNeighborsClassifier as knnC
    -   import pandas as pd
    -   from helper_packages.helpers import basicResults,makeTimingCurve
    -   from sklearn.pipeline import Pipeline
    -   from sklearn.preprocessing import StandardScaler
    -   from sklearn.ensemble import RandomForestClassifier
    -   from sklearn.feature_selection import SelectFromModel

    -   Execute the following command from the ANN directory:
        -   python ANN.py &
        -   Many .csv outputs will be provided that will be necessary when comparing
            performance across different parameters
        - Parameter Tuning:  By default, we specify .best_params_ for GridSearch to discover the best
            parameters.
            -   Make sure to comment & uncomment params_posture at every run.  This variable stores which variables
                to run a GridSearch on.  The first pass will find the parameters which output the best results.
                The second iteration should be set to a specification which you are curious about.  This will help tell
                the story of where/what to tune, and how this change impacts the results of the model

                -   Learning Curve: To analyze the different parameters and hyperparameters, make sure to hard code the
                    parameters/hyperparameters inside params_posture/params_approved based on what values
                    you'd like GridSearch to iterate.

                -   Model Complexity: Where applicable, one can use the parameters uncovered by .best_params_ and hard-code
                    the attributes to the <dataset>_final_params & <dataset>_OF_params variables to run:
                        -   makeTimingCurve
                        -   iterationLC
                    OF's main goal is to purposely OverFit the data, so choose hyperparameters that wouldn't surprise you if it
                    overfits.
    -   Graphing:
        -   Many .csv files will be spit out.  Use the <model>_<dataset>_LC_train & <model>_<dataset>_LC_test csv files to
            plot in Excel to plot the learning curves and model complexity curves.
            -   Where applicable, use the ITER csvs to plot the Hyperparameters, where iterations are telling of a story.

## SVM
### How to Run

-   Data will download automatically by the following command:
    -   csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00391/allUsers.lcl.csv')
-   Activate Python 3.6 with Required Libraries:
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

    -   Execute the following command from the ANN directory:
        -   python SVM.py &
        -   Many .csv outputs will be provided that will be necessary when comparing
            performance across different parameters
        - Parameter Tuning:  By default, we specify .best_params_ for GridSearch to discover the best
            parameters.
            -   Make sure to comment & uncomment params_posture at every run.  This variable stores which variables
                to run a GridSearch on.  The first pass will find the parameters which output the best results.
                The second iteration should be set to a specification which you are curious about.  This will help tell
                the story of where/what to tune, and how this change impacts the results of the model

                -   Learning Curve: To analyze the different parameters and hyperparameters, make sure to hard code the
                    parameters/hyperparameters inside params_posture/params_approved based on what values
                    you'd like GridSearch to iterate.

                -   Model Complexity: Where applicable, one can use the parameters uncovered by .best_params_ and hard-code
                    the attributes to the <dataset>_final_params & <dataset>_OF_params variables to run:
                        -   makeTimingCurve
                        -   iterationLC
                    OF's main goal is to purposely OverFit the data, so choose hyperparameters that wouldn't surprise you if it
                    overfits.
    -   Graphing:
        -   Many .csv files will be spit out.  Use the <model>_<dataset>_LC_train & <model>_<dataset>_LC_test csv files to
            plot in Excel to plot the learning curves and model complexity curves.
            -   Where applicable, use the ITER csvs to plot the Hyperparameters, where iterations are telling of a story.

## Boosting
### How to Run

-   Data will download automatically by the following command:
    -   csv = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00391/allUsers.lcl.csv')
-   Activate Python 3.6 with Required Libraries:
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

    -   Execute the following command from the ANN directory:
        -   python SVM.py &
        -   Many .csv outputs will be provided that will be necessary when comparing
            performance across different parameters
        - Parameter Tuning:  By default, we specify .best_params_ for GridSearch to discover the best
            parameters.
            -   Make sure to comment & uncomment params_posture at every run.  This variable stores which variables
                to run a GridSearch on.  The first pass will find the parameters which output the best results.
                The second iteration should be set to a specification which you are curious about.  This will help tell
                the story of where/what to tune, and how this change impacts the results of the model

                -   Learning Curve: To analyze the different parameters and hyperparameters, make sure to hard code the
                    parameters/hyperparameters inside params_posture/params_approved based on what values
                    you'd like GridSearch to iterate.

                -   Model Complexity: Where applicable, one can use the parameters uncovered by .best_params_ and hard-code
                    the attributes to the <dataset>_final_params & <dataset>_OF_params variables to run:
                        -   makeTimingCurve
                        -   iterationLC
                    OF's main goal is to purposely OverFit the data, so choose hyperparameters that wouldn't surprise you if it
                    overfits.
    -   Graphing:
        -   Many .csv files will be spit out.  Use the <model>_<dataset>_LC_train & <model>_<dataset>_LC_test csv files to
            plot in Excel to plot the learning curves and model complexity curves.
            -   Where applicable, use the ITER csvs to plot the Hyperparameters, where iterations are telling of a story.
