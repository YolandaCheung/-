#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics


clf = RandomForestClassifier(n_estimators=200,min_samples_split=50,
                             min_samples_leaf=20,max_depth=7,max_features='sqrt' ,random_state=10,oob_score=True)
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
pred=clf.predict(features_test)

from sklearn.metrics import accuracy_score
print accuracy_score(pred, labels_test)

print clf.oob_score_
print "AUC Score (Train): %f" % metrics.roc_auc_score(pred,labels_test)



#param_test1 = {'n_estimators':range(100,1000,100)}
#gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  #min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
                       #param_grid = param_test1, scoring='roc_auc',cv=5)
#gsearch1.fit(features_train,labels_train)




#param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
#gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, 
                                  #min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
   #param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
#gsearch2.fit(X,y)
#gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_



#print gsearch2.grid_scores_, 
#print gsearch2.best_params_, 
#print gsearch2.best_score_



try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
