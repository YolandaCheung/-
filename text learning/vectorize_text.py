#!/usr/bin/python
import os
import pickle
import re
import sys

sys.path.append( "L:/project/ud120-projects-master/tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("L:/project/ud120-projects-master/text_learning/from_sara.txt", "r")
from_chris = open("L:/project/ud120-projects-master/text_learning/from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:

        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset

        path = os.path.join('../', path[:-1])    
        email = open(path, "r")
        if name=="sara":
            from_data.append(0)
        if name=="chris":
            from_data.append(1)
        text3=parseOutText(email)
        t4=text3.replace("sara","")
        t5=t4.replace("shackleton","")
        t6=t5.replace("chris","")
        t7=t6.replace("germani","")
        word_data.append(t7)


            ### use parseOutText to extract the text from the opened email

            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]

            ### append the text to word_data

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris


        email.close()

print "emails processed"
print from_data

from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
### in Part 4, do TfIdf vectorization here
vectorizer = CountVectorizer()  
#计算个词语出现的次数  
X = vectorizer.fit_transform(word_data)  
#获取词袋中所有文本关键词  
word6 = vectorizer.get_feature_names()  
#查看词频结果  


transformer = TfidfTransformer()  
print transformer  
#将词频矩阵X统计成TF-IDF值  
tfidf = transformer.fit_transform(X)  
#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重  
