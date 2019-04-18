# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 09:55:38 2019

@author: 黃大祐
"""

from sklearn.datasets import fetch_20newsgroups
##也可以不另外抓分類
categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space']
trainingData = fetch_20newsgroups(subset = 'train',categories = categories)
#print(list(newsgroups_train.target_names))

#---------vectorized
#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
vectors_training = vectorizer.fit_transform(trainingData.data)
#--------------


#training process by multinomial bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
testData = fetch_20newsgroups(subset='test',categories = categories)
vectors_test = vectorizer.transform(testData.data)

model = MultinomialNB(alpha=.01)
model.fit(vectors_training,trainingData.target)

#test process
predicted = model.predict(vectors_test)
print("Macro-average: {0}".format(metrics.f1_score(testData.target,predicted,average='macro')))
print("Micro-average: {0}".format(metrics.f1_score(testData.target,predicted,average='micro')))
print(metrics.classification_report(testData.target,predicted))
print(metrics.confusion_matrix(testData.target, predicted))