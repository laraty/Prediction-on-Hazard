import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import RidgeCV
from sklearn.naive_bayes import GaussianNB
from datetime import datetime






X, Y, idx, testX, testidx = prepare_data()





DEVELOP = True
n_folds = 4
n_trees=50


clfs = [
        ExtraTreesRegressor(n_estimators = n_trees *2),
        RandomForestRegressor(n_estimators = n_trees),
        GradientBoostingRegressor(n_estimators = n_trees),
        KNeighborsClassifier(n_neighbors=3),
        KNeighborsRegressor(n_neighbors=3),
    ]


X_dev,Y_dev, X_test, Y_test,blend_train, blend_test, cv_results, cv_mse, cv_results_test=cvprediction(False,DEVELOP,n_folds, clfs, X,Y, testX)
pt=np.zeros((6,1))
for i in range(6):
    pt[i,0]=normalized_gin(Y_test, blend_test[:,i])
    

alphas = [0.0001, 0.005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
ensemble(3,alphas,blend_train[:,(0,1,2)], blend_test[:,(0,1,2)], Y_dev, Y_test, n_folds)


