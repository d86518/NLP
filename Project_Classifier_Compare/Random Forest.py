# -*- coding: utf-8 -*-
"""
@author: David Huang
Random forest (CART)
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn import ensemble
from sklearn import metrics
from sklearn.metrics import accuracy_score
pd.options.mode.chained_assignment = None

colnames = ['categorize', 'content']
news = pd.read_csv('news_files.csv',names=colnames,encoding = 'utf8')
news = news[1:len(news)]

#target = integer index of the category
news['target'] = pd.Series(0, index=news.index)
news_data =news.iloc[:6000]

#對原先df做slicing會造成SettingWithCopyWarning、用at填值
for i in range(1,len(news_data)+1):
    if news_data.iloc[i]['categorize']=='sports':
        news_data.at[i,'target'] = 0
    elif news_data.iloc[i]['categorize']=='politics':
        news_data.at[i,'target'] = 1
    else:
        news_data.at[i,'target'] = 2
        
#KFold(n_splits=’warn’, shuffle=False, random_state=None)
kfold = KFold(10, True)
predicted = []
expected = []

for train, test in kfold.split(np.arange(len(news_data))+1):
    news_train = news_data.iloc[train]
    news_test = news_data.iloc[test]
    
    # vectorized 
    vectorizer = TfidfVectorizer()
    vectors_training = vectorizer.fit_transform(news_train['content'].values.astype('U'))
    vectors_test = vectorizer.transform(news_test['content'].values.astype('U'))
    
    # 建立 random forest 模型 (default=gini)
    forest = ensemble.RandomForestClassifier(n_estimators = 100)
    forest.fit(vectors_training,news_train.target)
    
    #test process with NB
    #predicted = model.predict(vectors_test) Kfold need k times => change to array
    expected.extend(news_test.target)
    predicted.extend(forest.predict(vectors_test))

print("---------Random Forest---------")        
print("Macro-average: {0}".format(metrics.f1_score(expected,predicted,average='macro')))
print("Micro-average: {0}".format(metrics.f1_score(expected,predicted,average='micro')))
print(metrics.classification_report(expected,predicted))
print(metrics.confusion_matrix(expected, predicted))
accuracy = accuracy_score(expected, predicted)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

