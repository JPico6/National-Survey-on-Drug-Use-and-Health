# -*- coding: utf-8 -*-
"""
Created on Sat May 21 13:42:28 2016
@author: piccone
"""

from pandas import DataFrame
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from numpy import nan
from collections import Counter
from sklearn import preprocessing
import scipy.stats as stats
import random
from matplotlib import style 

random.seed( 190 )

print(os.getcwd())
df =  pd.read_csv("C:/Users/piccone/Desktop/DS projects/national survey on drug use and health 2012/ICPSR_34933/DS0001/Data.tsv", sep="\t")

##########Basic data exploration:##########
############################################################
pd.set_option('display.max_columns', 200) 
df.head(3)
df.shape
df.columns
df.describe()
df['CIGEVER'].unique()
df['CIGEVER'].value_counts()
df['CIGTRY'].value_counts()

##########Select Potentially Useful Features##########
############################################################

#subset all rows and the listed column names
df1 = pd.DataFrame(df)[['CIGEVER','CIGTRY','CIG30AV','SNFEVER','CHEWTRY','CIGAREVR','PIPEVER','ALCEVER',
          'ALCTRY','MJEVER','MJAGE','COCEVER','COCAGE','CRKEVER','CRKAGE','HEREVER','MTHAMP',
          'HERAGE','LSD','PCP','PEYOTE','MESC','PSILCY','ECSTASY','HALNOLST','HALLAGE','AMYLNIT','CLEFLU',
          'GAS','GLUE','ETHER','SOLVENT','LGAS','NITOXID','SPPAINT','AEROS','INHNOLST','INHAGE','DARVTYLC',
          'PERCTYLX', 'VICOLOR', 'HYDROCOD', 'METHDON', 'MORPHINE', 'OXYCONTN', 'ANLNOLST', 'ANALAGE', 
          'KLONOPIN', 'XNAXATVN', 'VALMDIAZ', 'TRNEVER', 'TRANAGE', 'METHDES', 'DIETPILS', 'RITMPHEN', 
          'STMNOLST', 'STMEVER', 'STIMAGE', 'SEDEVER','SEDAGE', 'ADDERALL', 'AMBIEN', 'COLDMEDS', 
          'KETAMINE', 'TRYPTMN', 'RSKPKCIG', 'RSKMJOCC', 'RKTRYLSD', 'RKTRYHER','RKHERREG',
          'RKCOCOCC','RK5ALDLY','RK5ALWK','RSKDIFMJ','RKDIFLSD','RKDIFCOC', 'RKDIFCRK',
          'RKDIFHER', 'RSKSELL', 'RKFQDNGR', 'RKFQRSKY', 'RKFQPBLT', 'RKFQDBLT','CIGIRTBL', 'CIGCRAVE',
          'CIGCRAGP', 'CIGINCTL', 'ALCCUTDN', 'ALCWD2SX', 'ALCPDANG', 'ALCFMFPB', 'MRJLIMIT','MRJCUTDN',
          'BOOKED', 'NOBOOKY2', 'PROBATON','DRVALDR', 'DRVAONLY', 'DRVDONLY', 'MMGETMJ', 'TXEVER', 'TXNDILAL', 'PREGNANT', 'NMERTMT2','INHOSPYR', 'LIFANXD', 'LIFASMA', 'LIFBRONC','LIFDEPRS',
          'LIFDIAB', 'LIFHARTD', 'LIFHBP', 'LIFHIV', 'LIFPNEU', 'LIFSTDS','LIFSINUS', 'LIFSLPAP', 'AUINPYR',
          'AUOPTYR', 'AURXYR', 'AUUNMTYR', 'SNMOV5Y2', 'SNYSELL','SNYSTOLE', 'SNYATTAK', 
          'SNFAMJEV','SNRLGSVC', 'SNRLGIMP', 'SNRLDCSN', 'SNRLFRND','YEMOV5Y2', 'YEATNDYR', 'YESCHFLT',
          'YESCHWRK', 'YESCHIMP', 'YESCHINT','YETCGJOB', 'YELSTGRD', 'YESTSCIG', 'YESTSMJ', 'YESTSALC',
          'YESTSDNK', 'YEPCHKHW', 'YEPHLPHW','YEPCHORE', 'YEPLMTTV', 'YEPLMTSN', 'YEPGDJOB','YEPPROUD',
          'YEYARGUP', 'YEYFGTSW', 'YEYFGTGP','YEYHGUN', 'YEYSELL', 'YEYSTOLE', 'YEYATTAK','YEPPKCIG',
          'YEPMJMO', 'YEPALDLY','YEGPKCIG', 'YEGMJEVR', 'YETLKNON', 'YETLKPAR', 'YETLKBGF','YETLKOTA',
          'YETLKSOP', 'YEPRTDNG','YEVIOPRV', 'YEDGPRGP', 'YESCHACT','YECOMACT', 'YEFAIACT', 'YEOTHACT',
          'YERLGSVC', 'YERLGIMP','YERLDCSN', 'YERLFRND', 'DSTNRV30', 'DSTHOP30','DSTRST30', 'DSTCHR30', 
          'DSTEFF30','DSTNGD30', 'DSTHOP12', 'DSTRST12','IMPRESP', 'SUICTHNK', 'SUICPLAN','SUICTRY', 
          'ADDPREV', 'ADDSCEV', 'ADLOSEV','ADDPDISC', 'ADWRSTHK', 'ADWRSPLN','ADWRSATP', 'AD_MDEA1',
          'AD_MDEA2', 'AD_MDEA3','AD_MDEA4', 'AD_MDEA5', 'AD_MDEA6', 'AD_MDEA7',
          'AD_MDEA8', 'AD_MDEA9', 'ADTMTNOW', 'ADRX12MO','YUHOSPYR', 'YUTPSTYR', 'YUSWSCYR', 'YODPREV',
          'YODSCEV', 'YOLOSEV', 'YODPDISC', 'YODPLSIN', 'YOWRDBTR', 'YOWRSTHK','YOWRSPLN',
          'YOWRSATP', 'YOSEEDOC', 'YORX12MO','CADRLAST', 'CADRPEOP', 'NRCH17_2','IRHHSIZ2',
          'IRKI17_2', 'IRHH65_2', 'IRFAMSZ2', 'IRKIDFA2','PRXYDATA', 'IRFAMSOC', 'IRFAMWAG', 'IRFSTAMP',
          'IRFAMSVC', 'IRPINC3', 'IRFAMIN3', 'MEDICARE', 'PRVHLTIN', 'GRPHLTIN', 'HLCNOTYR',
          'AGE2','NOMARR2', 'SERVICE', 'HEALTH', 'IRSEX', 'IRMARIT', 'IREDUC2', 'CATAG6', 'NEWRACE2',
          'SCHENRL', 'SDNTFTPT', 'EMPSTATY', 'PDEN00', 'COUTYP2']]

#save the smaller dataset to file          
df1.to_csv("C:/Users/piccone/Desktop/DS projects/national survey on drug use and health 2012/ICPSR_34933/DS0001/df1.csv", index=False)  

#load the smaller dataset
df1 = pd.read_csv("C:/Users/piccone/Desktop/DS projects/national survey on drug use and health 2012/ICPSR_34933/DS0001/df1.csv")

##########Examine features##########
############################################################

'''Significant recoding is necessary:

(a) Convert the following variables from 3 (logically assigned "yes") to 1 ("yes)'''
vars1 = ['LSD', 'PCP', 'PEYOTE', 'PSILCY', 'ECSTASY', 'AMYLNIT', 'CLEFLU', 'GAS', 'GLUE', 'ETHER', 'NITOXID',
    'AEROS', 'DARVTYLC', 'PERCTYLX', 'VICOLOR', 'HYDROCOD', 'METHDON', 'MORPHINE', 'OXYCONTN', 'KLONOPIN', 
    'METHDES', 'DIETPILS', 'RITMPHEN',  'MTHAMP', 'BOOKED']
for col in vars1:
    df1.loc[df1[col] == 3, col] = 1

'''(b) Convert 6 ("response not entered") = NaN'''
vars2 = ['MORPHINE', 'OXYCONTN', 'HYDROCOD', 'METHDON']
for col in vars2:
    df1.loc[df1[col] == 6, col] = np.nan
    

'''(c) Convert 4 ("No, logically assigned") =2 ("no")'''
vars3 = ['ANLNOLST', 'STMNOLST', 'MTHAMP'] 
for col in vars3:
    df1.loc[df1[col] == 4, col] = 2
  
'''(d) Convert 81 ("never used X type of drug, logically assigned" and 91 ("never used X type of drug") = 2 ("no")'''
vars4 = ['TRNEVER', 'STMEVER', 'SEDEVER'] 
for col in vars4:
    df1.loc[df1[col] == 81, col] = 2
for col in vars4:
    df1.loc[df1[col] == 91, col] = 2
 
'''(e) Convert 6 ("not entered") = 2 ("No"), everythng else = NaN'''
vars5 = ['LIFANXD', 'LIFASMA', 'LIFBRONC', 'LIFDEPRS', 'LIFDIAB', 'LIFHARTD', 'LIFHBP', 'LIFHIV', 'LIFPNEU',
     'LIFSTDS', 'LIFSINUS', 'LIFSLPAP', 'YETLKNON', 'YETLKPAR', 'YETLKBGF', 'YETLKOTA', 'YETLKSOP'] 
for col in vars5:
    df1.loc[df1[col] == 6, col] = 2

'''(f) Convert > 4 (multiple missing values) = NaN'''
vars6 = ['YELSTGRD', 'IMPRESP']
for col in vars6:
    df1.loc[df1[col] > 4, col] = np.nan
 
'''(g) Convert -9 ("missing") = NaN'''
df1['NRCH17_2'][df1.NRCH17_2 == -9] = np.nan

'''(h) Convert SCHENRL 3, 5, 11 ("Yes, logically assigned") = yes(1)'''
df1['SCHENRL'][df1.SCHENRL == 3] = 1
df1['SCHENRL'][df1.SCHENRL == 5] = 1
df1['SCHENRL'][df1.SCHENRL == 11] = 1

'''(i) Convert 91 ("Ever used") -> 2 ("no") '''
vars7 = ['CRKEVER', 'LSD', 'PCP', 'PEYOTE', 'MESC', 'PSILCY', 'ECSTASY', 'HALNOLST', 'AMYLNIT', 'CLEFLU', 'GAS', 'GLUE', 'ETHER',
  'SOLVENT', 'LGAS', 'NITOXID', 'SPPAINT', 'AEROS', 'INHNOLST', 'PERCTYLX', 'VICOLOR', 'HYDROCOD', 'METHDON', 'DARVTYLC',
  'MORPHINE', 'OXYCONTN', 'ANLNOLST', 'KLONOPIN', 'XNAXATVN', 'VALMDIAZ', 'TRNEVER', 'METHDES', 'DIETPILS', 
  'RITMPHEN', 'STMNOLST', 'STMEVER', 'SEDEVER']  
for col in vars7:
    df1.loc[df1[col] == 91, col] = 2  
  
'''Convert 81 ("Never used X type of drug")-> 2("No")'''
vars8 = ['METHDON', 'VICOLOR', 'HYDROCOD', 'PERCTYLX', 'DARVTYLC', 'MORPHINE', 'OXYCONTN', 'ANLNOLST', 'KLONOPIN', 'XNAXATVN',
    'VALMDIAZ', 'TRNEVER', 'METHDES', 'DIETPILS', 'RITMPHEN', 'STMNOLST', 'STMEVER', 'SEDEVER']      
for col in vars8:
    df1.loc[df1[col] == 81, col] = 2

'''(j) Convert all other 80+ to Nan'''
df1[df1 > 80] = np.nan

#save to file
df1.to_csv('df2.csv', index=False)  
df2 = pd.read_csv("C:/Users/piccone/Desktop/DS projects/national survey on drug use and health 2012/ICPSR_34933/DS0001/df2.csv")

'''A few plot examples for data exploration'''
style.use('ggplot')

counts = (Counter(df2['CIGEVER'])).values()
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('Yes','No')
y_pos = np.arange(len(objects))
plt.bar(y_pos, counts, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('Have You Ever Smoked a Cigarette?')

p = df1['COCEVER']
counts = Counter(p.values)

p = df2['COCEVER']
counts = Counter(p.values)
s2mask = np.isfinite(series2)

COCE = df2.COCEVER[np.isfinite(df2.COCEVER)]
counts = Counter(COCE).values()
fig = plt.figure()
ax1 = fig.add_subplot(111)
objects = ('Yes','No')
y_pos = np.arange(len(objects))
plt.bar(y_pos, counts, align='center', alpha=0.5, width=.5)
ax1.set_xticks([0,1])    
ax1.set_xticklabels(objects, rotation=0, fontsize=13)
ax1.set_title('Have You Ever Used Cocaine?')

#print distribution table
result = df2.apply(pd.value_counts).fillna(0); result

#print missing values in groups of 50 features
pd.isnull(df2.ix[:,0:50]).sum()

'''Standardize quantitative variables, and dummy code qualitative (where necessary). 
Also check data structure, are qualitative features set as such?'''

#split dataframe into quantitative data and categorical data
dfQ = df2.ix[:,['CIGTRY', 'CIG30AV', 'CHEWTRY', 'ALCTRY', 'ALCYRTOT', 'MJAGE', 'COCAGE', 'COCYRTOT', 'CRKAGE, HERAGE',  
'HALLAGE', 'INHAGE', 'ANALAGE', 'TRANAGE', 'STIMAGE', 'SEDAGE', 'MTHAAGE', 'RSKPKCIG', 'RSKMJOCC', 'RKTRYLSD', 
'RKTRYHER', 'RKHERREG', 'RKCOCOCC', 'RK5ALDLY', 'RK5ALWK', 'RSKDIFMJ', 'RKDIFLSD', 'RKDIFCOC', 'RKDIFCRK', 'RKDIFHER1', 
'RKFQDNGR', 'RKFQRSKY', 'RKFQPBLT', 'RKFQDBLT','CIGIRTBL', 'CIGCRAVE', 'CIGCRAGP', 'CIGINCTL', 
'NOBOOKY2', 'MMGETMJ',  'NMERTMT2', 'SNMOV5Y2', 'SNYSELL', 'SNYSTOLE', 'SNYATTAK', 'SNFAMJEV', 
'SNRLGSVC', 'SNRLGIMP', 'SNRLDCSN', 'SNRLFRND', 'YEMOV5Y2', 'YESCHFLT', 'YESCHWRK', 'YESCHIMP', 'YESCHINT', 'YETCGJOB',
'YELSTGRD', 'YESTSCIG', 'YESTSMJ', 'YESTSALC', 'YESTSDNK', 'YEPCHKHW', 'YEPHLPHW', 'YEPCHORE', 'YEPLMTTV', 
'YEPLMTSN', 'YEPGDJOB','YEPPROUD','YEYARGUP', 'YEYFGTSW', 'YEYFGTGP','YEYHGUN', 'YEYSELL', 
'YEYSTOLE', 'YEYATTAK','YEPPKCIG','YEPMJMO', 'YEPALDLY','YEGPKCIG', 'YEGMJEVR',
'YESCHACT', 'IMPRESP', 'YESCHACT','YECOMACT', 'YEFAIACT', 'YEOTHACT',
'YERLGSVC', 'YERLGIMP','YERLDCSN', 'YERLFRND', 'DSTNRV30', 'DSTHOP30','DSTRST30', 'DSTCHR30', 
'DSTEFF30','DSTNGD30', 'DSTHOP12', 'DSTRST12','IMPRESP', 
'CADRLAST','CADRPEOP','NRCH17_2', 'IRHHSIZ2', 'IRKI17_2', 'IRHH65_2', 'IRFAMSZ2', 'IRKIDFA2', 
'IRPINC3', 'IRFAMIN3', 'AGE2', 'HEALTH', 'IREDUC2']]

dfC = df2.ix[:,['CIGEVER','SNFEVER','CIGAREVR','PIPEVER','ALCEVER','MJEVER','COCEVER','CRKEVER',
'HEREVER','LSD','PCP','PEYOTE','MESC','PSILCY','ECSTASY','HALNOLST','AMYLNIT','CLEFLU',
'GAS','GLUE','ETHER','SOLVENT','LGAS','NITOXID','SPPAINT','AEROS','INHNOLST','DARVTYLC',
'PERCTYLX', 'VICOLOR', 'HYDROCOD', 'METHDON', 'MORPHINE', 'OXYCONTN', 'ANLNOLST',  
'KLONOPIN', 'XNAXATVN', 'VALMDIAZ', 'TRNEVER', 'METHDES', 'DIETPILS', 'RITMPHEN', 
'STMNOLST', 'STMEVER', 'SEDEVER','ADDERALL', 'AMBIEN', 'COLDMEDS', 'KETAMINE', 
'TRYPTMN', 'RSKSELL', 'ALCCUTDN', 'ALCWD2SX', 'ALCPDANG', 'ALCFMFPB', 'MRJLIMIT','MRJCUTDN',
'BOOKED', 'PROBATON','DRVALDR', 'DRVAONLY', 'DRVDONLY', 'TXEVER', 'TXNDILAL', 'PREGNANT', 'INHOSPYR', 'LIFANXD', 'LIFASMA', 'LIFBRONC','LIFDEPRS',
'LIFDIAB', 'LIFHARTD', 'LIFHBP', 'LIFHIV', 'LIFPNEU', 'LIFSTDS','LIFSINUS', 'LIFSLPAP', 'AUINPYR',
'AUOPTYR', 'AURXYR', 'AUUNMTYR', 'YEATNDYR', 'YETLKNON', 'YETLKPAR', 'YETLKBGF',
'YETLKOTA','YETLKSOP', 'YEPRTDNG','YEVIOPRV', 'YEDGPRGP', 'SUICTHNK', 'SUICPLAN','SUICTRY', 
'ADDPREV', 'ADDSCEV', 'ADLOSEV','ADDPDISC', 'ADWRSTHK', 'ADWRSPLN','ADWRSATP', 'AD_MDEA1',
'AD_MDEA2', 'AD_MDEA3','AD_MDEA4', 'AD_MDEA5', 'AD_MDEA6', 'AD_MDEA7',
'AD_MDEA8', 'AD_MDEA9', 'ADTMTNOW', 'ADRX12MO','YUHOSPYR', 'YUTPSTYR', 'YUSWSCYR', 'YODPREV',
'YODSCEV', 'YOLOSEV', 'YODPDISC', 'YODPLSIN', 'YOWRDBTR', 'YOWRSTHK','YOWRSPLN',
'YOWRSATP', 'YOSEEDOC', 'YORX12MO','PRXYDATA', 'IRFAMSOC', 'IRFAMWAG', 'IRFSTAMP',
'IRFAMSVC', 'MEDICARE', 'PRVHLTIN', 'GRPHLTIN', 'HLCNOTYR', 
'NOMARR2', 'SERVICE', 'IRSEX', 'SCHENRL', 'SDNTFTPT']]

'''using the preprocessing.scale works, but only with data with no missing values: 
dfQ_scaled = preprocessing.scale(dfQ), therefore, create empty data frame and go:'''

dfQZ = pd.DataFrame()

for i in dfQ.transpose():
    dfQZ.loc[:,i] = (dfQ.ix[:,i] - dfQ.ix[:,i].mean())/dfQ.ix[:,i].std(ddof=0)

dfQZ.columns = ['CIGTRY', 'CIG30AV', 'CHEWTRY', 'ALCTRY', 'ALCYRTOT', 'MJAGE', 'COCAGE', 'COCYRTOT', 'CRKAGE, HERAGE',  
'HALLAGE', 'INHAGE', 'ANALAGE', 'TRANAGE', 'STIMAGE', 'SEDAGE', 'MTHAAGE', 'RSKPKCIG', 'RSKMJOCC', 'RKTRYLSD', 
'RKTRYHER', 'RKHERREG', 'RKCOCOCC', 'RK5ALDLY', 'RK5ALWK', 'RSKDIFMJ', 'RKDIFLSD', 'RKDIFCOC', 'RKDIFCRK', 'RKDIFHER1', 
'RKFQDNGR', 'RKFQRSKY', 'RKFQPBLT', 'RKFQDBLT','CIGIRTBL', 'CIGCRAVE', 'CIGCRAGP', 'CIGINCTL', 
'NOBOOKY2', 'MMGETMJ',  'NMERTMT2', 'SNMOV5Y2', 'SNYSELL', 'SNYSTOLE', 'SNYATTAK', 'SNFAMJEV', 
'SNRLGSVC', 'SNRLGIMP', 'SNRLDCSN', 'SNRLFRND', 'YEMOV5Y2', 'YESCHFLT', 'YESCHWRK', 'YESCHIMP', 'YESCHINT', 'YETCGJOB',
'YELSTGRD', 'YESTSCIG', 'YESTSMJ', 'YESTSALC', 'YESTSDNK', 'YEPCHKHW', 'YEPHLPHW', 'YEPCHORE', 'YEPLMTTV', 
'YEPLMTSN', 'YEPGDJOB','YEPPROUD','YEYARGUP', 'YEYFGTSW', 'YEYFGTGP','YEYHGUN', 'YEYSELL', 
'YEYSTOLE', 'YEYATTAK','YEPPKCIG','YEPMJMO', 'YEPALDLY','YEGPKCIG', 'YEGMJEVR',
'YESCHACT', 'IMPRESP', 'YESCHACT','YECOMACT', 'YEFAIACT', 'YEOTHACT',
'YERLGSVC', 'YERLGIMP','YERLDCSN', 'YERLFRND', 'DSTNRV30', 'DSTHOP30','DSTRST30', 'DSTCHR30', 
'DSTEFF30','DSTNGD30', 'DSTHOP12', 'DSTRST12','IMPRESP', 
'CADRLAST','CADRPEOP','NRCH17_2', 'IRHHSIZ2', 'IRKI17_2', 'IRHH65_2', 'IRFAMSZ2', 'IRKIDFA2', 
'IRPINC3', 'IRFAMIN3', 'AGE2', 'HEALTH', 'IREDUC2']

#dfQZ['CIGTRY'] = (df2.CIGTRY - df2.CIGTRY.mean())/df2.CIGTRY.std(ddof=0)

'''dummy code the categorical variables - does it work properly (ie. 2 level var = 1 var?)
dfC_D = dummy coded...'''
  
MARITAL = pd.get_dummies(df2['IRMARIT'])
MARITAL.columns = ['MARRIED','WIDOWED','DIVORCED','NEVER_MARRIED']

RACE = pd.get_dummies(df2['NEWRACE2'])
RACE.columns = ['WHITE','BLACK','NATIVEAM','PACISL','ASIAN','MULTIPLE','HISPANIC']

EMPLOY = pd.get_dummies(df2['EMPSTATY'])
EMPLOY.columns =['FULLTIME','PARTTIME','UNEMPLOYED','OTHER']

AGECAT = pd.get_dummies(df2['CATAG6'])
AGECAT.columns =['AGE12_17','AGE18_25','AGE26_34','AGE35_49','AGE50_64','AGE65']

COUNTY = pd.get_dummies(df2['COUTYP2'])
COUNTY.columns = ['COUNTY_LARGE','COUNTY_SMALL','COUNTY_NONMETRO']

dfC = dfC.replace(1,0)
dfC = dfC.replace(2,1)

#recombind dataset
df3 = pd.concat([dfQZ, dfC, MARITAL,RACE,EMPLOY,AGECAT,COUNTY], axis=1)

#check/remove collinearity  (highly correlated variables)

DataFrame.corr(method='pearson', min_periods=1)

dfQZ.ix[:,11:20].corr(method='pearson',min_periods=1)
dfC.ix[:,1:10].corr(method='pearson',min_periods=1)
#not as bad as I feared

#missing values
#count number of missing per feature
pd.isnull(df2.ix[:,240:260]).sum()

'''most of the variables have huge missing data because they are questions about use of 
drugs that many people did not use - so main analyses should be based on whether 
people used various drugs as predictors, additional analysis should look at particular 
types of users. Also, some questions are designed for just adults, others just for kids '''

'''df1.fillna(df1.mean())
df1.fillna(df1.mean()['B':'C'])
df.dropna(axis=0)   #drop columns with missing data

also this: from sklearn.preprocessing import Imputer'''

df3.to_csv('df3.csv', index=False)  

