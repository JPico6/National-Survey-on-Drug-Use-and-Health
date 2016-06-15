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

#####################################Basic data exploration:
pd.set_option('display.max_columns', 200) 
df.head(3)
df.shape
df.columns
df.describe()
df['CIGEVER'].unique()
df['CIGEVER'].value_counts()
df['CIGTRY'].value_counts()

'''...and so on...

#################################Select Potentially Useful Features'''

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

##########################Examine features       

'''Significant recoding is necessary:
(1) Convert some values to missing
(2) Other extraneous recoding

(a)
convert LSD, PCP, PEYOTE, PSILCY, ECSTASY, AMYLNIT, CLEFLU, GAS, GLUE, ETHER, NITOXID,
    AEROS, DARVTYLC, PERCTYLX, VICOLOR, HYDROCOD, METHDON, MORPHINE, OXYCONTN, KLONOPIN 
    METHDES, DIETPILS, RITMPHEN,  MTHAMP, BOOKED, values of 3 (logically assigned "yes") to 1 ("yes")
'''
df1['LSD'][df1.LSD == 3] = 1
df1['PCP'][df1.PCP == 3] = 1
df1['PEYOTE'][df1.PEYOTE == 3] = 1
df1['PSILCY'][df1.PSILCY == 3] = 1
df1['AMYLNIT'][df1.AMYLNIT == 3] = 1
df1['CLEFLU'][df1.CLEFLU == 3] = 1
df1['GAS'][df1.GAS == 3] = 1
df1['GLUE'][df1.GLUE == 3] = 1
df1['ETHER'][df1.ETHER == 3] = 1
df1['NITOXID'][df1.NITOXID == 3] = 1
df1['AEROS'][df1.AEROS == 3] = 1
df1['DARVTYLC'][df1.DARVTYLC == 3] = 1
df1['PERCTYLX'][df1.PERCTYLX == 3] = 1
df1['VICOLOR'][df1.VICOLOR == 3] = 1
df1['HYDROCOD'][df1.HYDROCOD == 3] = 1
df1['METHDON'][df1.METHDON == 3] = 1
df1['MORPHINE'][df1.MORPHINE == 3] = 1
df1['OXYCONTN'][df1.OXYCONTN == 3] = 1
df1['KLONOPIN'][df1.KLONOPIN == 3] = 1
df1['METHDES'][df1.METHDES == 3] = 1
df1['DIETPILS'][df1.DIETPILS == 3] = 1
df1['RITMPHEN'][df1.RITMPHEN == 3] = 1
df1['MTHAMP'][df1.MTHAMP == 3] = 1
df1['BOOKED'][df1.BOOKED == 3] = 1

'''(b)
MORPHINE, OXYCONTN, HYDROCOD, METHDON    6 ("response not entered") = NaN'''
df1['MORPHINE'][df1.MORPHINE == 6] = np.nan
df1['OXYCONTN'][df1.OXYCONTN == 6] = np.nan
df1['HYDROCOD'][df1.HYDROCOD == 6] = np.nan
df1['METHDON'][df1.METHDON == 6] = np.nan

'''(c)
ANLNOLST, STMNOLST, MTHAMP1  4 ("No, logically assigned") =2 ("no")'''
df1['ANLNOLST'][df1.ANLNOLST == 4] = 2
df1['STMNOLST'][df1.STMNOLST == 4] = 2
df1['MTHAMP'][df1.MTHAMP == 4] = 2

'''(d)*******************************
TRNEVER, STMEVER, SEDEVER  81 ("never used X type of drug, logically assigned" and 91 ("never used X type of drug") = 2 ("no")'''
df1['TRNEVER'][df1.TRNEVER == 81] = 2
df1['TRNEVER'][df1.TRNEVER == 91] = 2
df1['STMEVER'][df1.STMEVER == 81] = 2
df1['STMEVER'][df1.STMEVER == 91] = 2
df1['SEDEVER'][df1.SEDEVER == 81] = 2
df1['SEDEVER'][df1.SEDEVER == 91] = 2

'''(e)
LIFANXD, LIFASMA, LIFBRONC, LIFDEPRS, LIFDIAB, LIFHARTD, LIFHBP, LIFHIV, LIFPNEU,
     LIFSTDS, LIFSINUS, LIFSLPAP, YETLKNON1, YETLKPAR1, YETLKBGF1, YETLKOTA1, YETLKSOP1, 6 ("not entered") = 2 ("No"), everythng else = NaN'''
df1['LIFANXD'][df1.LIFANXD == 6] = 2
df1['LIFASMA'][df1.LIFASMA == 6] = 2
df1['LIFBRONC'][df1.LIFBRONC == 6] = 2
df1['LIFDEPRS'][df1.LIFDEPRS == 6] = 2
df1['LIFDIAB'][df1.LIFDIAB == 6] = 2
df1['LIFHARTD'][df1.LIFHARTD == 6] = 2
df1['LIFHBP'][df1.LIFHBP == 6] = 2
df1['LIFHIV'][df1.LIFHIV == 6] = 2
df1['LIFPNEU'][df1.LIFPNEU == 6] = 2
df1['LIFSTDS'][df1.LIFSTDS == 6] = 2
df1['LIFSINUS'][df1.LIFSINUS == 6] = 2
df1['LIFSLPAP'][df1.LIFSLPAP == 6] = 2
df1['YETLKNON'][df1.YETLKNON == 6] = 2
df1['YETLKPAR'][df1.YETLKPAR == 6] = 2
df1['YETLKBGF'][df1.YETLKBGF == 6] = 2
df1['YETLKOTA'][df1.YETLKOTA == 6] = 2
df1['YETLKSOP'][df1.YETLKSOP == 6] = 2

'''(f)
YELSTGRD and IMPRESP > 4 (multiple missing values) = NaN'''
df1['YELSTGRD'][df1.YELSTGRD > 4] = np.nan
df1['IMPRESP'][df1.IMPRESP > 4] = np.nan

'''(g)
NRCH17_2 -9 ("missing") = NaN'''
df1['NRCH17_2'][df1.NRCH17_2 == -9] = np.nan

'''(h)
SCHENRL 3, 5, 11 ("Yes, logically assigned") = yes(1)'''
df1['SCHENRL'][df1.SCHENRL == 3] = 1
df1['SCHENRL'][df1.SCHENRL == 5] = 1
df1['SCHENRL'][df1.SCHENRL == 11] = 1

'''(i)
Change a few others to no...
CRKEVER, LSD, PCP, PEYOTE, MESC, PSILCY, ECSTASY, HALNOLST, AMYLNIT, CLEFLU, GAS, GLUE, ETHER,
  SOLVENT, LGAS, NITOXID, SPPAINT, AEROS, INHNOLST, PERCTYLX, VICOLOR, HYDROCOD, METHDON, DARVTYLC,
  MORPHINE, OXYCONTN, ANLNOLST, KLONOPIN, XNAXATVN, VALMDIAZ, TRNEVER, METHDES, DIETPILS, 
  RITMPHEN, STMNOLST, STMEVER, SEDEVER,  91 ("Ever used") -> 2 ("no") '''
  
df1['CRKEVER'][df1.CRKEVER == 91] = 2
df1['LSD'][df1.LSD == 91] = 2
df1['PCP'][df1.PCP == 91] = 2
df1['PEYOTE'][df1.PEYOTE == 91] = 2
df1['MESC'][df1.MESC == 91] = 2
df1['PSILCY'][df1.PSILCY == 91] = 2
df1['ECSTASY'][df1.ECSTASY == 91] = 2
df1['HALNOLST'][df1.HALNOLST == 91] = 2
df1['AMYLNIT'][df1.AMYLNIT == 91] = 2
df1['CLEFLU'][df1.CLEFLU == 91] = 2
df1['GAS'][df1.GAS == 91] = 2
df1['GLUE'][df1.GLUE == 91] = 2
df1['ETHER'][df1.ETHER == 91] = 2
df1['SOLVENT'][df1.SOLVENT == 91] = 2
df1['LGAS'][df1.LGAS == 91] = 2
df1['NITOXID'][df1.NITOXID == 91] = 2
df1['SPPAINT'][df1.SPPAINT == 91] = 2
df1['AEROS'][df1.AEROS == 91] = 2
df1['INHNOLST'][df1.INHNOLST == 91] = 2
df1['PERCTYLX'][df1.PERCTYLX == 91] = 2
df1['VICOLOR'][df1.VICOLOR == 91] = 2
df1['HYDROCOD'][df1.HYDROCOD == 91] = 2
df1['METHDON'][df1.METHDON == 91] = 2
df1['DARVTYLC'][df1.DARVTYLC == 91] = 2
df1['MORPHINE'][df1.MORPHINE == 91] = 2
df1['OXYCONTN'][df1.OXYCONTN == 91] = 2
df1['ANLNOLST'][df1.ANLNOLST == 91] = 2
df1['KLONOPIN'][df1.KLONOPIN == 91] = 2
df1['XNAXATVN'][df1.XNAXATVN == 91] = 2
df1['VALMDIAZ'][df1.VALMDIAZ == 91] = 2
df1['TRNEVER'][df1.TRNEVER == 91] = 2
df1['METHDES'][df1.METHDES == 91] = 2
df1['DIETPILS'][df1.DIETPILS == 91] = 2
df1['RITMPHEN'][df1.RITMPHEN == 91] = 2
df1['STMNOLST'][df1.STMNOLST == 91] = 2
df1['STMEVER'][df1.STMEVER == 91] = 2
df1['SEDEVER'][df1.SEDEVER == 91] = 2

'''METHDON, VICOLOR, HYDROCOD, PERCTYLX, DARVTYLC, MORPHINE, OXYCONTN, ANLNOLST, KLONOPIN, XNAXATVN,
    VALMDIAZ, TRNEVER, METHDES, DIETPILS, RITMPHEN, STMNOLST, STMEVER, SEDEVER,      81 ("Never used X type of drug")-> 2("No")'''
df1['METHDON'][df1.METHDON == 81] = 2
df1['VICOLOR'][df1.VICOLOR == 81] = 2
df1['HYDROCOD'][df1.HYDROCOD == 81] = 2
df1['PERCTYLX'][df1.PERCTYLX == 81] = 2
df1['DARVTYLC'][df1.DARVTYLC == 81] = 2
df1['MORPHINE'][df1.MORPHINE == 81] = 2
df1['OXYCONTN'][df1.OXYCONTN == 81] = 2
df1['ANLNOLST'][df1.ANLNOLST == 81] = 2
df1['KLONOPIN'][df1.KLONOPIN == 81] = 2
df1['XNAXATVN'][df1.XNAXATVN == 81] = 2
df1['VALMDIAZ'][df1.VALMDIAZ == 81] = 2
df1['TRNEVER'][df1.TRNEVER == 81] = 2
df1['METHDES'][df1.METHDES == 81] = 2
df1['DIETPILS'][df1.DIETPILS == 81] = 2
df1['RITMPHEN'][df1.RITMPHEN == 81] = 2
df1['STMNOLST'][df1.STMNOLST == 81] = 2
df1['STMEVER'][df1.STMEVER == 81] = 2
df1['SEDEVER'][df1.SEDEVER == 81] = 2

'''(j)
Convert all other 80+ to Nan'''
df1[df1 > 80] = np.nan

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

'''###############check/remove collinearity  (highly correlated variables)'''

DataFrame.corr(method='pearson', min_periods=1)

dfQZ.ix[:,11:20].corr(method='pearson',min_periods=1)
dfC.ix[:,1:10].corr(method='pearson',min_periods=1)
#not as bad as I feared

'''##############################missing values'''
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




