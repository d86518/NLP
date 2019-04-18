# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 10:37:12 2019

@author: 黃大祐
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
model = Pipeline([('tfidf',TfidfVectorizer()),('clf',MultinomialNB(alpha=.01))])
trainingData = fetch_20newsgroups(subset = 'train')
testData = fetch_20newsgroups(subset='test')


model.fit(trainingData.data,trainingData.target)

#test process
predicted = model.predict(testData.data)
print("Macro-average: {0}".format(metrics.f1_score(testData.target,predicted,average='macro')))
print("Micro-average: {0}".format(metrics.f1_score(testData.target,predicted,average='micro')))
print(metrics.classification_report(testData.target,predicted))
print(metrics.confusion_matrix(testData.target, predicted))