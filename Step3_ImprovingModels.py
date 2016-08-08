# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 13:45:47 2016

@author: piccone

Improving on the base random forest model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation as cv
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import ensemble
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import random
from matplotlib import style 
from sklearn import metrics
import math
from sklearn import svm

style.use('ggplot')

random.seed(190)

##########prepare data##########
############################################################
dfAdult2 = pd.read_csv("C:/Users/piccone/Desktop/DS projects/national survey on drug use and health 2012/ICPSR_34933/DS0001/dfAdult2.csv")

msk = np.random.rand(len(dfAdult2)) < 0.75
train = dfAdult2[msk]
test = dfAdult2[~msk]

trainX = train.drop(['BOOKED'], axis=1)
testX = test.drop(['BOOKED'], axis=1)

trainY = train['BOOKED']
testY = test['BOOKED']

#test weighting for stacked ensemble (this section of code is adapted from Julian, 2016):
def vclas(w1,w2,w3, w4, w5):
    Xtrain,Xtest, ytrain,ytest= cv.train_test_split(trainX,trainY,test_size=0.4)

    clf1 = LogisticRegression()
    clf2 = GaussianNB()
    clf3 = RandomForestClassifier(n_estimators=10,bootstrap=True)
    clf4= ExtraTreesClassifier(n_estimators=10, bootstrap=True)
    clf5 = GradientBoostingClassifier(n_estimators=10)

    clfes=[clf1,clf2,clf3,clf4, clf5]

    eclf = VotingClassifier(estimators=[('lr', clf1), ('gnb', clf2), ('rf', clf3),('et',clf4), ('gb',clf5)],
                            voting='soft',
                            weights=[w1, w2, w3,w4, w5])

    [c.fit(Xtrain, ytrain) for c in (clf1, clf2, clf3,clf4, clf5, eclf)]
 
    N = 6
    ind = np.arange(N)
    width = 0.3
    fig, ax = plt.subplots()

    for i, clf in enumerate(clfes):
        print(clf,i)
        p1=ax.bar(i,clfes[i].score(Xtrain,ytrain,), width=width,color="blue", alpha=0.5)
        p2=ax.bar(i+width,clfes[i].score(Xtest,ytest,), width=width,color="red", alpha=0.5)
    ax.bar(len(clfes)+width,eclf.score(Xtrain,ytrain,), width=width,color="blue", alpha=0.5)
    ax.bar(len(clfes)+width *2,eclf.score(Xtest,ytest,), width=width,color="red", alpha=0.5)
    plt.axvline(4.8, color='k', linestyle='dashed')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(['LogisticRegression',
                        'GaussianNB',
                        'RandomForestClassifier',
                        'ExtraTrees',
                        'GradientBoosting',
                        'VotingClassifier'],
                       rotation=40,
                       ha='right')
    plt.title('Training and Test Score for Different Classifiers')
    plt.legend([p1[0], p2[0]], ['training', 'test'], loc='lower left')
    plt.show()

vclas(2,1.5,2,2,2)

##########Fit final model##########
######################################################################
clf1 = LogisticRegression(random_state=123)
clf2 = GaussianNB()
clf3 = RandomForestClassifier(n_estimators=100, random_state=123)
clf4= ExtraTreesClassifier(n_estimators=10, bootstrap=True,random_state=123)
clf5 = GradientBoostingClassifier(n_estimators=10, random_state=123)

clf1.fit(trainX, trainY)
clf2.fit(trainX, trainY)
clf3.fit(trainX, trainY)
clf4.fit(trainX, trainY)
clf5.fit(trainX, trainY)

eclf = VotingClassifier(estimators=[('lr', clf1), ('gnb', clf2), ('rf', clf3),('et',clf4),('gb',clf5)],
                            voting='soft',
                            weights=[2, 1.5, 2, 2, 2])
eclf1 = eclf.fit(trainX, trainY)
disbursed = eclf1.predict_proba(testX)
fpr, tpr, thresholds = metrics.roc_curve(testY, disbursed[:,1])
metrics.auc(fpr, tpr) 
#.830

'''Let's try using stacked generalization for overall health'''
#use two training sets
from sklearn.cross_validation import train_test_split
train, train2 = train_test_split(dfAdult2, train_size = 0.4)
train2, test = train_test_split(train2, train_size = 0.65)

trainX = train.drop(['HEALTH'], axis=1)
train2X = train2.drop(['HEALTH'], axis=1)
testX = test.drop(['HEALTH'], axis=1)

trainY = train['HEALTH']
train2Y = train2['HEALTH']
testY = test['HEALTH']

AllTrainX = trainX.append(train2X)
AllTrainY = trainY.append(train2Y)

#Establish models:
c1 = svm.SVR()
c2 = linear_model.Lasso(alpha=0.1)
c3 = RandomForestRegressor(n_estimators=100,bootstrap=True, random_state=123)
c4=  ExtraTreesRegressor(n_estimators=10, bootstrap=True,random_state=123)
c5 = ensemble.GradientBoostingRegressor(n_estimators=10, random_state=123)

#fit models on a
c1.fit(trainX, trainY)
c2.fit(trainX, trainY)
c3.fit(trainX, trainY)
c4.fit(trainX, trainY)
c5.fit(trainX, trainY)

#predict models on b
c1p = c1.predict(train2X)
c2p = c2.predict(train2X)
c3p = c3.predict(train2X)
c4p = c4.predict(train2X)
c5p = c5.predict(train2X)

#fit models on b
c1.fit(train2X, train2Y)
c2.fit(train2X, train2Y)
c3.fit(train2X, train2Y)
c4.fit(train2X, train2Y)
c5.fit(train2X, train2Y)

#predict models on a
c1pa = c1.predict(trainX)
c2pa = c2.predict(trainX)
c3pa = c3.predict(trainX)
c4pa = c4.predict(trainX)
c5pa = c5.predict(trainX)

#combine the predictions into new data set with actual y values
trainAT = pd.DataFrame([c1pa,c2pa,c3pa,c4pa,c5pa,trainY]).transpose()
trainBT = pd.DataFrame([c1p,c2p,c3p,c4p,c5p,train2Y]).transpose()
trainALL = trainAT.append(trainBT)
trainALL.columns = ['c1p','c2p','c3p','c4p','c5p','y']

TotalX = trainALL.drop(['y'], axis=1)
TotalY = trainALL['y']

#train predictions with linear meta-regressor
lr = LinearRegression()
lr.fit(TotalX, TotalY)

preds = lr.predict(TotalX)
mse = mean_squared_error(TotalY, preds)
#.778
math.sqrt(mse)
#.882  - very marginal improvement, let's try a random forest meta-regressor

rf = RandomForestRegressor(n_estimators=100,bootstrap=True, random_state=123)
rf.fit(TotalX, TotalY)
preds= rf.predict(TotalX)
mse = mean_squared_error(TotalY, preds)
#.116
math.sqrt(mse)
#.340  #huge improvement

#test it on the test set
c1.fit(AllTrainX, AllTrainY)
c2.fit(AllTrainX, AllTrainY)
c3.fit(AllTrainX, AllTrainY)
c4.fit(AllTrainX, AllTrainY)
c5.fit(AllTrainX, AllTrainY)

#predict models on test
c1t = c1.predict(testX)
c2t = c2.predict(testX)
c3t = c3.predict(testX)
c4t = c4.predict(testX)
c5t = c5.predict(testX)

testAT = pd.DataFrame([c1t,c2t,c3t,c4t,c5t,testY]).transpose()
testAT.columns = ['c1p','c2p','c3p','c4p','c5p','y']

TotalTestX = testAT.drop(['y'], axis=1)
TotalTestY = testAT['y']

#stack with linear meta-regressor
preds = lr.predict(TotalTestX)
mse = mean_squared_error(TotalTestY, preds)
#.778
math.sqrt(mse)
#.882  (compared with .889)

#stack with rf meta-regressor
preds= rf.predict(TotalTestX)
mse = mean_squared_error(TotalTestY, preds)
#.8177
math.sqrt(mse)
#.904  

'''plot religious services by health. I group them below, for ease of presentation,
into two groups, the healthy group (who responded that they are excellent, very good, or good)
and the unhealthy group (who responded fair or poor)'''

df2 = pd.read_csv("C:/Users/piccone/Desktop/DS projects/national survey on drug use and health 2012/ICPSR_34933/DS0001/df2.csv")

'''separate out just adults (participants over 17 years old)'''
dfAdult = df2.loc[df3['AGE12_17'] != 1]

df4 = np.column_stack((dfAdult['HEALTH'],dfAdult['SNRLGSVC']))
df5 = pd.DataFrame(df4)
df5 = df5.dropna(axis=0)
df5.columns = ['Health','RServices']

len(df5['RServices'][df5.Health < 3])
len(df5['RServices'][df5.Health > 3])

n_groups = 6
y1 = (df5['RServices'][df5.Health < 3].value_counts())/len(df5[df5.Health < 3])
y2 = (df5['RServices'][df5.Health > 3].value_counts())/len(df5[df5.Health > 3])
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.4
rects1 = plt.bar(index, y1, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Healthy')
rects2 = plt.bar(index + bar_width, y2, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Unhealthy')
plt.xlabel('Times Attended Religious Services Past Year')
plt.ylabel('% of Health Group')
plt.title('Religious Service Attendance by Health Ratings')
plt.xticks(index + bar_width, ('Never', '1-2', '3-5', '6-24', '25-52','52+'))
plt.legend()
plt.tight_layout()
plt.show()

'''Less healthy people are less likely to attend religious services. 
