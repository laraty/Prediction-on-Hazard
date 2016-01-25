from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV
from sklearn import metrics



def ensemble(Method,alphas,blend_train, blend_test, Y_dev, Y_test, n_folds):
   if (Method==1):
        bclf = RidgeCV(alphas=alphas, normalize=True, cv=n_folds)
        bclf.fit(blend_train, Y_dev)
        print ("Best alpha = ", bclf.alpha_)
        Y_test_predict = bclf.predict(blend_test)
   elif(Method==2):
        bclf = ElasticNetCV(alphas=alphas, normalize=True, cv=n_folds)
        bclf.fit(blend_train, Y_dev)
        print ("Best alpha = ", bclf.alpha_)
        Y_test_predict = bclf.predict(blend_test)
   else:
        bclf = LassoCV(alphas=alphas, normalize=True, cv=n_folds)
        bclf.fit(blend_train, Y_dev)
        print ("Best alpha = ", bclf.alpha_)
        Y_test_predict = bclf.predict(blend_test)
        
   score1 = metrics.mean_absolute_error(Y_test, Y_test_predict)
   score = normalized_gini(Y_test, Y_test_predict)
    
   return score1, score
