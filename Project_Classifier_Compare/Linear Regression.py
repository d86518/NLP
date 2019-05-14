import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score  
from sklearn.metrics import r2_score
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn import metrics


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
    
    lm = LinearRegression()
    lm.fit(vectors_training,news_train.target)
    expected.extend(news_test.target)
    predicted.extend(lm.predict(vectors_test))

#np.float64 -> native float
for i in range(len(expected)):
    expected[i] = expected[i].item()
    predicted[i] = predicted[i].item()

#可能有缺值(nan) -> 用前後一個數平均
for i in range(len(expected)):
    if str(expected[i]) == 'nan':
        expected[i] = (expected[i-1]+expected[i+1])/2
    if str(predicted[i]) == 'nan':
        predicted[i] = (predicted[i-1]+predicted[i+1])/2

#預測的數字非整數->round/無條件捨去都可
for i in range(len(predicted)):
    predicted[i]  = round(predicted[i])
    
print("---------LinearRegression---------")      
print("explained_variance_score: ",explained_variance_score(expected, predicted))
print("mean_absolute_error: ",mean_absolute_error(expected, predicted))
print("mean_squared_error: ",mean_squared_error(expected, predicted))
print("median_absolute_error: ",median_absolute_error(expected, predicted))
print("r2_score: ",r2_score(expected, predicted))
accuracy = accuracy_score(expected, predicted)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(metrics.classification_report(expected,predicted))

#from gensim.models.word2vec import Word2Vec
#from sklearn.decomposition import PCA
#vectors_training = Word2Vec(news_train['content'].values.astype('U'))
#vectors_training = PCA(n_components=2).fit_transform(vectors_training)
