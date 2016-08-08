# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:58:37 2016

@author: piccone

Analysis for adults

"""

from pandas import DataFrame
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from numpy import nan
from collections import Counter
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import random
from sklearn.metrics import mean_squared_error
import math
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from matplotlib import style 

style.use('ggplot')

random.seed(190)

##########prepare data##########
############################################################
df3 = pd.read_csv("C:/Users/piccone/Desktop/DS projects/national survey on drug use and health 2012/ICPSR_34933/DS0001/df3.csv")

'''separate out just adults (participants over 17 years old)'''
dfAdult = df3.loc[df3['AGE12_17'] != 1]

'''
How many missing per variable? I examined this in groups of 50'''
pd.isnull(dfAdult1.ix[:,0:50]).sum()
pd.isnull(dfAdult1.ix[:,51:100]).sum()
pd.isnull(dfAdult1.ix[:,101:150]).sum()
pd.isnull(dfAdult1.ix[:,151:200]).sum()
pd.isnull(dfAdult1.ix[:,201:251]).sum()

dfAdult1 = dfAdult.ix[:,['RSKPKCIG', 'RSKMJOCC', 'RKTRYLSD', 'RKTRYHER', 
'RKCOCOCC', 'RK5ALWK', 'RSKDIFMJ', 'RKDIFLSD', 'RKDIFCOC', 'RKDIFCRK', 'RKFQDNGR', 'RKFQRSKY', 'RKFQPBLT', 'RKFQDBLT', 'NMERTMT2', 'SNYSELL', 'SNYSTOLE', 'SNYATTAK',
'SNFAMJEV', 'SNRLGSVC', 'SNRLGIMP', 'SNRLDCSN', 'SNRLFRND', 'DSTNRV30', 'DSTHOP30', 'DSTRST30',
'DSTCHR30', 'DSTEFF30', 'DSTNGD30', 'IRHHSIZ2', 'IRKI17_2', 'IRHH65_2', 'IRFAMSZ2', 'IRKIDFA2',
'IRPINC3', 'IRFAMIN3', 'AGE2', 'HEALTH', 'IREDUC2', 'CIGEVER', 'SNFEVER', 'CIGAREVR', 'PIPEVER',
'ALCEVER', 'MJEVER', 'COCEVER', 'CRKEVER', 'HEREVER', 'PCP', 'PEYOTE', 'MESC', 'PSILCY', 'ECSTASY',
'HALNOLST', 'AMYLNIT', 'CLEFLU', 'GAS', 'GLUE', 'ETHER', 'SOLVENT', 'LGAS', 'NITOXID', 'SPPAINT',
'AEROS', 'INHNOLST', 'DARVTYLC', 'PERCTYLX', 'ANLNOLST', 'KLONOPIN', 'XNAXATVN', 'VALMDIAZ', 
'TRNEVER', 'METHDES', 'DIETPILS', 'RITMPHEN', 'STMNOLST', 'STMEVER', 'SEDEVER', 'ADDERALL',
'AMBIEN', 'COLDMEDS', 'KETAMINE', 'RSKSELL', 'BOOKED', 'PROBATON', 'TXEVER', 'TXNDILAL', 'INHOSPYR',
'AUINPYR', 'AUOPTYR', 'AURXYR', 'AUUNMTYR', 'SUICTHNK', 'ADDPREV', 'IRFAMSOC', 'IRFAMWAG', 'IRFAMSVC',
'MEDICARE', 'PRVHLTIN', 'HLCNOTYR', 'SERVICE', 'IRSEX', 'SCHENRL', 
'MARRIED', 'WIDOWED', 'DIVORCED', 'NEVER_MARRIED', 'WHITE', 'BLACK', 'PACISL', 'ASIAN', 'MULTIPLE',
'HISPANIC', 'FULLTIME', 'PARTTIME', 'UNEMPLOYED', 'OTHER','AGE18_25','AGE26_34','AGE35_49',
'AGE50_64','AGE65','COUNTY_LARGE','COUNTY_SMALL','COUNTY_NONMETRO']]

dfAdult2 = dfAdult1.dropna(axis=0)
dfAdult2.to_csv("C:/Users/piccone/Desktop/DS projects/national survey on drug use and health 2012/ICPSR_34933/DS0001/dfAdult2.csv", index=False)  

'''#plot distributions of outcome variables'''

#Respondent been arrested?
counts = Counter(dfAdult2.BOOKED).values()
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('Yes','No')
y_pos = np.arange(len(objects))
plt.bar(y_pos, counts, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('Have You Ever Been Arrested?')

'''obviously far more people have not been booked, but this is still a fine variable
to test on, as many cases include variable values that are infrequent.'''

#Respondent overall health
ax=plt.subplot(111)
plt.hist(dfAdult2.HEALTH, bins=5, alpha=0.5)  
ax.set_xticklabels(('Excellent', 'Very Good', 'Good','Fair', 'Poor'), fontsize=8)
ax.set_xticks([-0.80,0.10,0.90,1.80,2.70])
plt.title('Distribution of Overall Health Ratings')
plt.xlabel("Health Rating", fontsize=16)  
plt.ylabel("Count", fontsize=16)

'''positively skewed - most people are relatively healthy, with a few trailing off
to fair and poor values.'''

'''explore demographic information:'''
#Respondent Sex
counts = Counter(dfAdult2.IRSEX).values()
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('Male','Female')
y_pos = np.arange(len(objects))
plt.bar(y_pos, counts, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('What is Your Gender?')

#Respondent Arrests
counts = Counter(dfAdult2.BOOKED).values()
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('Yes','No')
y_pos = np.arange(len(objects))
plt.bar(y_pos, counts, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('Have You Ever Been Arrested?')

#Respondent marital status
cMarried = list(Counter(dfAdult2.MARRIED).values())[1]
cWidowed = list(Counter(dfAdult2.WIDOWED).values())[1]
cDivorced = list(Counter(dfAdult2.DIVORCED).values())[1]
cNeverM = list(Counter(dfAdult2.NEVER_MARRIED).values())[1]
tMARRIED = [cMarried,cWidowed,cDivorced,cNeverM]
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('Married','Widowed','Divorced','Never Married')
y_pos = np.arange(len(objects))
plt.bar(y_pos, tMARRIED, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1,2,3])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('What is Your Marital Status?')

#Respondent age
c18_25 = list(Counter(dfAdult2.AGE18_25).values())[1]
c26_34 = list(Counter(dfAdult2.AGE26_34).values())[1]
c35_49 = list(Counter(dfAdult2.AGE35_49).values())[1]
c50_64 = list(Counter(dfAdult2.AGE50_64).values())[1]
c65 = list(Counter(dfAdult2.AGE65).values())[1]
tAGE = [c18_25,c26_34,c35_49,c50_64,c65]
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('18-25','26-34','35-49','50-64','65+')
y_pos = np.arange(len(objects))
plt.bar(y_pos, tAGE, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1,2,3,4])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('What is Your Age?')

#Respondent county size
cLarge = list(Counter(dfAdult2.COUNTY_LARGE).values())[1]
cMedium = list(Counter(dfAdult2.COUNTY_SMALL).values())[1]
cSmall = list(Counter(dfAdult2.COUNTY_NONMETRO).values())[1]
tMETRO = [cLarge,cMedium,cSmall]
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('Large','Medium','Small')
y_pos = np.arange(len(objects))
plt.bar(y_pos, tMETRO, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1,2])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('What Type of County do you Live in?')

#Respondent ethnicity
cWhite = list(Counter(dfAdult2.WHITE).values())[1]
cBlack = list(Counter(dfAdult2.BLACK).values())[1]
cPacisl = list(Counter(dfAdult2.PACISL).values())[1]
cAsian = list(Counter(dfAdult2.ASIAN).values())[1]
cHispanic = list(Counter(dfAdult2.HISPANIC).values())[1]
cOther = list(Counter(dfAdult2.MULTIPLE).values())[1]
tETHNICITY = [cWhite, cBlack, cPacisl, cAsian, cHispanic, cOther]
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('White','Black','Pac.Isl','Asian','Hispanic','Other')
y_pos = np.arange(len(objects))
plt.bar(y_pos, tETHNICITY, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1,2,3,4,5])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('What Is Your Ethnicity?')

#Respondent employment status
cFull = list(Counter(dfAdult2.FULLTIME).values())[1]
cPart = list(Counter(dfAdult2.PARTTIME).values())[1]
cUne = list(Counter(dfAdult2.UNEMPLOYED).values())[1]
cOther = list(Counter(dfAdult2.OTHER).values())[1]
tEMPLOY = [cFull, cPart, cUne, cOther]
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('Full-Time','Part-Time','Unemployed','Other')
y_pos = np.arange(len(objects))
plt.bar(y_pos, tEMPLOY, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1,2,3])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('What Is Your Employment Status?')

#Respondent sex
cSEX = list(Counter(dfAdult2.IRSEX).values())
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('Male','Female')
y_pos = np.arange(len(objects))
plt.bar(y_pos, cSEX, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('What Is Your Sex?')

#Respondent school status
cSCHOOL = list(Counter(dfAdult2.SCHENRL).values())
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('Yes','No')
y_pos = np.arange(len(objects))
plt.bar(y_pos, cSCHOOL, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('Are You Currently Enrolled in Any School?')

#Respondent income
ax=plt.subplot(111)
plt.hist(dfAdult2.IRPINC3, bins=7, alpha=0.5)  
ax.set_xticklabels(('< 10k', '10-19k','20-29k','30-39k','40-49k','50-75k','75k+'), fontsize=8)
ax.set_xticks([-0.40,0.10,0.60,1.085,1.60,2.10,2.60])
plt.title("Respondent's Income Level")
#plt.xlabel("Income", fontsize=16)  
#plt.ylabel("Count", fontsize=16)

'''divide sample into train and test'''
msk = np.random.rand(len(dfAdult2)) < 0.75
train = dfAdult2[msk]
test = dfAdult2[~msk]

'''Let's try random forest predicting if they've ever been booked (arrested). Booked is just an 
arbitrary outcome, and random forests are a great, quick, and powerful way to set
a benchmark for which to compare further models.'''
trainX = train.drop(['BOOKED'], axis=1)
testX = test.drop(['BOOKED'], axis=1)

trainY = train['BOOKED']
testY = test['BOOKED']

rf = RandomForestClassifier(n_estimators=100)
%timeit rf.fit(trainX, trainY)

#how accurate on the train set
disbursed = rf.predict_proba(trainX)
fpr, tpr, thresholds = metrics.roc_curve(trainY, disbursed[:,1])
metrics.auc(fpr, tpr)         #calculate roc using the trapezoidal rule
#1.00

#plot auc
plt.plot(fpr,tpr)
plt.xlabel('False Positives')
plt.ylabel('True Positives')
plt.title('AUC: Have You Ever Been Arrested-Training Set')

#how accurate on the test set
disbursed = rf.predict_proba(testX)
fpr, tpr, thresholds = metrics.roc_curve(testY, disbursed[:,1])
metrics.auc(fpr, tpr)         #calculate roc using the trapezoidal rule
#.8224
#plot auc
plt.plot(fpr,tpr)
plt.xlabel('False Positives')
plt.ylabel('True Positives')
plt.title('AUC: Have You Ever Been Arrested-Test Set')

disbursed = rf.predict(testX)     
confusion_matrix(disbursed, testY)
#.856 accuracy

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cm = confusion_matrix(testY, disbursed)
np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
plt.figure()
plot_confusion_matrix(cm)
'''we can see we predict people who have not been booked very well, but we also predict
too many people to not be booked (quite a few false negatives). '''

# Normalize the confusion matrix by row (i.e by the number of samples
# in each class)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

'''variable importance for the above RF'''
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
for f in range(trainX.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

top10 = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), trainX.columns), 
             reverse=True)[:10]
top10 = pd.DataFrame(top10)

objects = top10[1]
plt.bar(range(top10.shape[0]), top10[0], align="center", alpha=0.5)
plt.xticks(range(10), objects, rotation='vertical')
plt.title('Feature Importance Predicting Respondent Arrests')

#test on overall health
trainX = train.drop(['HEALTH'], axis=1)
testX = test.drop(['HEALTH'], axis=1)

trainY = train['HEALTH']
testY = test['HEALTH']

rf = RandomForestRegressor(n_estimators=100)
%timeit rf.fit(trainX, trainY)

#predict on the test set:
p1 = rf.predict(testX)
mse = mean_squared_error(testY, p1)
#.790
math.sqrt(mse)
#.889

'''variable importance for the above RF'''
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
for f in range(trainX.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

top10 = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), trainX.columns), 
             reverse=True)[:10]
top10 = pd.DataFrame(top10)

objects = top10[1]
plt.bar(range(top10.shape[0]), top10[0], align="center", alpha=0.5)
plt.xticks(range(10), objects, rotation='vertical')
plt.title('Feature Importance Predicting Respondent Health')

'''One alternative is to perform a principle component analysis (PCA) to combine features. This
will be especially effective if the components that emerge are actually interpretable, which
is not gaurantted! 

We could also explore these variables in greater detail. For example, TXEVER (ever received
treatment for drugs/alcohol) could be further explored with the feature AUMOTVYR,
which is what prompted people to get treatment for past mental health issues -- such as whether
they did so voluntarily or not. But it's unclear if people who were forced to receive 
mental health treatment where treated for drug/alcohol abuse or not.'''

##########PCA ANALYSIS##########
############################################################

trainX = train.drop(['BOOKED'], axis=1)
testX = test.drop(['BOOKED'], axis=1)

trainY = train['BOOKED']
testY = test['BOOKED']

#How many components is ideal? Let's first test how much variance is accounted for:
# Plot the PCA spectrum
pca = decomposition.PCA()
pca.fit(trainX)
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.title('Components by Variance Explained', fontsize=13)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')

#After 22 components, each additional component explains less than .5% of the total variance.
#However, the total variance explained increases up to 95 components. We can try both.
n_components = [94,95,96]
Cs = np.logspace(-4, 4, 3)
logistic = linear_model.LogisticRegression()

pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
estimator = GridSearchCV(pipe,
                         dict(pca__n_components=n_components,
                              logistic__C=Cs))
estimator.fit(trainX, trainY)

plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
            linestyle=':', label='n_components chosen')
plt.legend(prop=dict(size=12))
plt.show()

#95 is the optimal number of components

# fits PCA, transforms data and fits the decision tree classifier
# on the transformed data
pipe = Pipeline([('pca', PCA(n_components=95)),
                 ('tree', RandomForestClassifier(n_estimators=100))])
pipe.fit(trainX, trainY)

disbursed = pipe.predict_proba(testX)
fpr, tpr, thresholds = metrics.roc_curve(testY, disbursed[:,1])
metrics.auc(fpr, tpr) 
#AUC drops to .797

'''trying this analysis for the health data:'''
trainX = train.drop(['HEALTH'], axis=1)
testX = test.drop(['HEALTH'], axis=1)

trainY = train['HEALTH']
testY = test['HEALTH']

pipe = Pipeline([('pca', PCA(n_components=95)),
                 ('tree', RandomForestRegressor(n_estimators=100))])
pipe.fit(trainX, trainY)

disbursed = pipe.predict(testX)
mse = mean_squared_error(testY, disbursed)
#.834
math.sqrt(mse)
#.913
'''theoretically using PCA could improve the model's predictive ability as it may
reduced overfitting - by reducing the number of features in an orthogonal manner,
we were able to remove variables which just added noise to the train model - so the random
forest trained on that noise in the benchmark model. In addition, we gain power through
the reduction of dimensionality. Nevertheless, PCA depricated the model slightly.
