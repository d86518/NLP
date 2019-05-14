# -*- coding: utf-8 -*-
"""
Created on Sun May  5 01:34:33 2019

@author: 黃大祐
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
pd.options.mode.chained_assignment = None

colnames = ['categorize', 'content']
news = pd.read_csv('news_files.csv',names=colnames,encoding = 'utf8')
news = news[1:len(news)]

news['target'] = pd.Series(0, index=news.index)
for i in range(1,len(news)+1):
    if i<=3000:
        news.at[i,'target'] = 0 #sport
    elif 3000<i<=5000:
        news.at[i,'target'] = 1 #politics
    else:
        news.at[i,'target'] = 2 #health
        
predicted = []
expected = []

model = XGBClassifier(n_jobs=-1)
kfold = KFold(10, True)
vectorizer = TfidfVectorizer()

for train, test in kfold.split(news):
    news_train = news.iloc[train]
    news_test = news.iloc[test]
    
    X_train = vectorizer.fit_transform(news_train['content'].values.astype('U'))
    Y_train = news_train.target

    X_test = vectorizer.transform(news_test['content'].values.astype('U'))

    model.fit(X_train, Y_train)  #開始train
    predicted.extend(model.predict(X_test)) #測資預測
    expected.extend(news_test.target) #比對用正解 Y_test = news_test.target


print("---------XGBoost---------")      
accuracy = accuracy_score(expected, predicted)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(metrics.classification_report(expected,predicted))


"""
使用sklearn cross_val_score -> unsuccessful
from sklearn.model_selection import cross_val_score
X = vectorizer.fit_transform(news_data.content.values.astype('U'))
Y = news_data.target
results = cross_val_score(model, X, Y, cv=10, scoring='accuracy', n_jobs=-1)
print(results)
print(results.mean())    
"""