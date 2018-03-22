#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
 


import pickle
import sys
sys.path.append("L:/project/ud120-projects-master/tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("L:/project/ud120-projects-master/final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = 'L:/project/ud120-projects-master/tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)

### it's all yours from here forward!  
from sklearn import cross_validation
#X_train, X_test, Y_train, Y_test= cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train,test in kf:   
    features_train=[features[ii] for ii in train]
    features_test=[features[ii] for ii in test]
    labels_train=[labels[jj] for jj in train]
    labels_test=[labels[jj] for jj in test]
    
from sklearn.model_selection import GridSearchCV
#parameter={'min_samples_split ':[5.0,15.0]}
tree_para = {'criterion':['gini','entropy'],'min_samples_split':[2,3,4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,100,120,150]}
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf=GridSearchCV(clf, tree_para, cv=5)
clf = clf.fit(features_train,labels_train)
pred=clf.predict(features_test)


from sklearn.metrics import accuracy_score
print accuracy_score(pred, labels_test)