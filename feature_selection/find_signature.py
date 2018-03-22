#!/usr/bin/python
 
import pickle
import numpy
import numpy as np
numpy.random.seed(42)
 
 
### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "C:/Users/Administrator/your_word_data.pkl"
authors_file = "C:/Users/Administrator/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )
 
 
 
### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)
 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')

features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test)
 
### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150]
labels_train   = labels_train[:150]
 
 
from sklearn.feature_selection import SelectPercentile, f_classif
 
selector= SelectPercentile(f_classif, percentile=10)
selector.fit(features_train, labels_train)
 
features_train=selector.transform(features_train)
features_test=selector.transform(features_test)
 
print "no. of Chris tranning emails:", sum(labels_train)
print "no. of Sara tranning emails:", len(labels_train)-sum(labels_train)
 
 




#import sys
#from class_vis import prettyPicture, output_image
#from prep_terrain_data import makeTerrainData
 
#import matplotlib.pyplot as plt
#import numpy as np
#import pylab as pl
#from classifyDT import classify
#features_train, labels_train, features_test, labels_test = makeTerrainData()
#clf = classify(features_train, labels_train)
#prettyPicture(clf, features_test, labels_test)
#output_image("test.png", "png", open("test.png", "rb").read())
 
from sklearn import linear_model
from sklearn.linear_model import Lasso
 
     

lassoRegression = linear_model.Lasso()
lassoRegression=lassoRegression.fit(features_train,labels_train)

print("权重向量:%s, b的值为:%.2f" % (lassoRegression.coef_, lassoRegression.intercept_))
print len(lassoRegression.coef_)
print("损失函数的值:%.2f" % np.mean((lassoRegression.predict(features_test) - labels_test) ** 2))
print("预测性能得分: %.2f" % lassoRegression.score(features_test,labels_test))
 
from sklearn import tree
clf=tree.DecisionTreeClassifier()
clf=clf.fit(features_train,labels_train)
pred=clf.predict(features_test,labels_test)
print max(clf.feature_importances_)
print len(clf.feature_importances_)


from sklearn.metrics import accuracy_score
print accuracy_score(pred, labels_test)

