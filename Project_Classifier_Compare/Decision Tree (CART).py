# -*- coding: utf-8 -*-
"""
Created on Tue May 14 20:58:44 2019

@author: 黃大祐
"""
from sklearn import tree
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import accuracy_score

colnames = ['categorize', 'content']
news = pd.read_csv('news_files.csv',names=colnames,encoding = 'utf8')
news = news[1:len(news)]

#target = integer index of the category
news['target'] = pd.Series(0, index=news.index)
news_data =news.iloc[:6000]

#對原先df做slicing會造成SettingWithCopyWarning、用at填值
for i in range(len(news_data)+1):
    if news_data.iloc[i]['categorize']=='sports':
        news_data.at[i,'target'] = 0
    elif news_data.iloc[i]['categorize']=='politics':
        news_data.at[i,'target'] = 1
    else:
        news_data.at[i,'target'] = 2
        
#做10-folds cross validation 
#KFold(n_splits=’warn’, shuffle=False, random_state=None)
kfold = KFold(10, True)
predicted = []
expected = []
for train, test in kfold.split(np.arange(len(news_data))+1):
    news_train = news_data.iloc[train]
    news_test = news_data.iloc[test]
    vectorizer = TfidfVectorizer()
    
    vectors_training = vectorizer.fit_transform(news_train['content'].values.astype('U'))
    vectors_test = vectorizer.transform(news_test['content'].values.astype('U'))
    
    model = tree.DecisionTreeClassifier() #不調參數 default = CART
    model.fit(vectors_training,news_train.target)
    
    # predict
    expected.extend(news_test.target)
    predicted.extend(model.predict(vectors_test))
        
#Decision Tree
print("---------Decision Tree---------")
print("Macro-average: {0}".format(metrics.f1_score(expected,predicted,average='macro')))
print("Micro-average: {0}".format(metrics.f1_score(expected,predicted,average='micro')))
print(metrics.classification_report(expected,predicted,target_names = ['sports','politics','health'] ))
accuracy = accuracy_score(expected, predicted)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
