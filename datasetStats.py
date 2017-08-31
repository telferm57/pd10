# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 11:00:22 2017

@author: M Telfer
"""

#%% audio stats 
audioDownloaded = set()
#basic stats : 

#Numeric fiedls in demogs 
import synapse1 as syn1
#imp.reload(synapse1)
global demogs 
global scores 

syn1.setGlobals()
syn1.initDatasets()
syn1.syn1.voiceDf.shape
demogs.shape
# --- data cleanliness     
 
demogs['medication-start-year'].isnull().sum() 
# set invalid years to nan 
demogs['medication-start-year'] = demogs['medication-start-year'].apply(lambda x: np.nan if x<1960 else x)
demogs['medication-start-year'].describe()
demogs['medication-start-year'].isnull().sum()
demogs['onset-year'].describe()
demogs[demogs['onset-year']< 1950].count()
demogs['onset-year'].isnull().sum()
demogs['diagnosis-year'].describe()
demogs[demogs['diagnosis-year']< 1950].shape
demogs['diagnosis-year'].isnull().sum()
demogs['age'].describe()
demogs[demogs['age']< 10].shape
demogs['age'].isnull().sum()
demogs['packs-per-day'].describe()
demogs[demogs['packs-per-day']< 10].shape
demogs['packs-per-day'].isnull().sum()

sum(demogs['years-smoking']==0)
demogs[demogs['years-smoking']< 10].shape
demogs['years-smoking'].isnull().sum()
demogs['years-smoking'] = demogs['years-smoking'].apply(lambda x: np.nan if x==0 else x)
demogs['years-smoking'].describe()
demogs['last-smoked'].describe()
demogs.dtypes
demogs['heathHistory']= demogs['health-history'].astype('category')
demogs['heathHistory'].describe()

print('No. of unique patients in demogs files: ',len(set(demogs.healthCode)))   


demogs.columns
tt = demogs.describe()
type(tt)
# --- Number of voice files per HC 
syn1.syn1.voiceDf.healthCode.nunique()
bins=[0,2,4,6,10,15,100]
syn1.syn1.voiceDf.healthCode.value_counts().median()
syn1.syn1.voiceDf.healthCode.value_counts().hist(bins=bins)
type(syn1.syn1.voiceDf.healthCode.value_counts())
vc = syn1.syn1.voiceDf.healthCode.value_counts().values

# --- create set information  create sets of health codes for various demographics 
voicehc = set(syn1.voiceDf.healthCode)
demohc = set(demogs.healthCode)
scoreshc  = set(scores.healthCode)
PwPdiagnosed = set(demogs.healthCode[demogs['professional-diagnosis']==True])
demogs['professional-diagnosis'].value_counts()
len(PwPdiagnosed)

print('number submitting a score survey',len(scoreshc))
 
scores.shape
scores.columns
scoresdupe= scores[scores.duplicated(subset='healthCode',keep=False)]
scd = scoresdupe.copy()
scd.sort_values('healthCode',inplace=True)
scd.head()
scd.shape
scd.drop_duplicates(inplace=True)
voiceWithHC = set.intersection(voicehc,demohc)
voiceWithScore = set.intersection(voicehc,scoreshc,demohc)
voiceComplete = set.intersection(voicehc,scoreshc,demohc)
voiceCompletePwP = set.intersection(voicehc,scoreshc,demohc,PwPdiagnosed)
306/5718
# --- general stats
# --- smoking 
for cc in demogs.columns:
    print(demogs[cc].value_counts())
    
print('Number indiviuals in demographic survey',len(demohc))
print('Number diagnosed with PwP',len(PwPdiagnosed))
demogs.gender.value_counts().Male/len(demogs.gender)
print('proportion Male: ', demogs.gender.value_counts().Male/len(demogs.gender))

print('proportion Female')
demogs.gender.value_counts().Female/len(demogs.gender)
demogs[demogs.healthCode.isin(PwPdiagnosed)].gender.value_counts().Male/len(PwPdiagnosed)
demogs[demogs.healthCode.isin(PwPdiagnosed)].gender.value_counts()
# scores information - who is submitting them ? 

demogs.columns
demogs.rename(columns={'deep-brain-stimulation':'dbs'},inplace=True)
PwPwithDBS = demogs[demogs.healthCode.isin(PwPdiagnosed)].dbs.value_counts()[True]
print('Number of PwP with DBS', PwPwithDBS)

demogs.rename(columns={'medication-start-year':'msy'},inplace=True)
PwPmsy = demogs[demogs.healthCode.isin(PwPdiagnosed)][demogs.msy>1950].shape
print('Number of PwP taking medication ', PwPmsy[0])

print('Total number of people who have submitted a updrs survey :',len(scoreshc))


print('Total num diagnosed pwp submitting survey with voice files: ',len(voiceCompletePwP))

voicePwP = syn1.voiceDf[syn1.voiceDf.healthCode.isin(PwPdiagnosed)]

print('Total num  voice files submitted by PwP: ',len(voicePwP))

voicePwPScore = syn1.voiceDf[syn1.voiceDf.healthCode.isin(PwPdiagnosed) &
                        syn1.voiceDf.healthCode.isin(scoreshc)]

print('''Total num  voice files submitted by
      PwP who have also submitted updrs survey : ''',voicePwPScore.shape[0])

''' how can we get number of voice files submitted within one hour of UPDRS score ? ''' 
for cl in df.columns:
    print(cl,':')
vca = np.array(voiceComplete)
vca = list(voiceComplete)
len(vca)
vca[1]
len(voiceWithHC)
len(voiceWithScore)
len(vc[vc>20])
len(vc[vc>40])
plt.hist(vc,bins)
#analysis of voicefiles that have a demographic entry : 
# age, gender, 
dfcomplete = demogs.join(syn1.voiceDf.set_index('healthCode'),
                         on='healthCode',
                         rsuffix='dd',how='inner')
dfcomplete.drop_duplicates(inplace=True,subset='healthCode')
dfcomplete.columns
dfcomplete['professional-diagnosis'].value_counts()
dfcomplete.age.hist(bins=[20,30,40,50,60,70,80,90,100])
dfcomplete[:5]
dfpwp = scores.join(dfcomplete.set_index('healthCode'),
                     on='healthCode',
                     rsuffix='dd',how='inner')
dfpwp.shape
dfpwp.columns
dfpwp['MDS-UPDRS2.1'].hist(bins=[0,1,2,3,4])
vc = dfpwp['MDS-UPDRS2.1'].value_counts().sort_index()
vc.plot(kind='bar')
vc
dfpwp.age.hist(bins=[20,30,40,50,60,70,80,90,100])
dfpwp['professional-diagnosis'].value_counts()
dfpwp[dfpwp['professional-diagnosis']==True]['MDS-UPDRS2.1'].value_counts()

#--- updrs scores for those with professional diagnosis 
dfcomplete['scoresind']=dfcomplete['healthCode'].isin(scoreshc)
dfcomplete[dfcomplete.scoresind==False].age.value_counts()
dfcomplete['agerange']=pd.cut(dfcomplete.age,[1,20,30,40,50,60,70,80,90,100])
dfcomplete[dfcomplete.scoresind==True].agerange.value_counts()

# --- how many pwp have submitted updrs score ? 
demogs['inscores'] = demogs.healthCode.isin(scoreshc)
dfd = demogs[(demogs['professional-diagnosis'] == True) & (demogs.inscores==True)]

# --- age plot 
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# -- plot age for PwPs 

demogs[demogs.healthCode.isin(PwPdiagnosed)]

fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Number')
ax[0].set_title(r'Age profile of PwP')
demogs[demogs.healthCode.isin(PwPdiagnosed)].age.hist(bins=10,ax=ax[0])
ax[1].set_xlabel('Age')
ax[1].set_ylabel('Number')
ax[1].set_title(r'Age profile of non-PwP')
demogs[~demogs.healthCode.isin(PwPdiagnosed)].age.hist(bins=10,ax=ax[1])
# Tweak spacing to prevent clipping of ylabel
#fig.tight_layout()
plt.show()

# age plot of sample
PwPSamplehc = set(df.healthCode[df.diagnosed==True] )
nPwPSamplehc = set(df.healthCode[df.diagnosed!=True] ) 
fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
ax[0].set_xlabel('Age')
ax[0].set_ylabel('Number')
ax[0].set_title(r'Age profile of PwP')
demogs[demogs.healthCode.isin(PwPSamplehc)].age.hist(bins=10,ax=ax[0])
ax[1].set_xlabel('Age')
ax[1].set_ylabel('Number')
ax[1].set_title(r'Age profile of non-PwP')
demogs[demogs.healthCode.isin(nPwPSamplehc)].age.hist(bins=10,ax=ax[1])
# Tweak spacing to prevent clipping of ylabel
#fig.tight_layout()
plt.show()
#dfd.age.hist(by=demogs['professional-diagnosis'])
#dfd.age.describe()
#fig, axes = plt.subplots(ncols=2, figsize=(12, 5), sharey=True,sharex=True)
#dfd.boxplot(column='age', ax=axes[0])
#nodiag.boxplot(column='age', ax=axes[1])
#plt.show()

type(nodiag)
diagTrue  = demogs[(demogs['professional-diagnosis'] == True)]
print('Number with professional diagnosis = yes ',diagTrue.shape[0])
print('number with professional diag & have submitted survey:', dfd[0],dfd[0]/diagTrue )

# --- how many pwp have voice file? 



syn1.setFileAvailabilityInd(syn1.walkDf,syn1.voiceDf) # (adds voice and walk datafile indicator )
syn1.demogs.columns
nulldiag= demogs[demogs['professional-diagnosis'].isnull()]
nulldiaghc = set(nulldiag.healthCode)

nodiag = demogs[(demogs['professional-diagnosis'] == False) ]
nodiag.age.hist()
nodiag.age.describe()
nodiag.boxplot(column='age', ax=ax1)


print('number with no prof diag but have submitted UPDRS:', 
      len(set.intersection(nodiaghc,scoreshc)) )


dfd = demogs[(demogs['professional-diagnosis'] == False) & (demogs['onset-year']>1900)]
print('number with no prof diag but do have onset year:', dfd.shape[0] )
print('number with no prof diag but do have onset year who have submitted score:',
                  len(set(dfd.healthCode).intersection(scoreshc)))
print('number with null prof diag who have submitted score:',
                  len(set.intersection(scoreshc,nulldiaghc)))

dfd = demogs[(demogs['professional-diagnosis'] == False) & (demogs.msy >1970)]
print('number with no prof diag but do have medical start year :', dfd.shape[0] )



onsetSurvey = len(dfd.healthCode.isin(scoreshc))
print('number with no prof diag but do have onset year who have submitted survey:', onsetSurvey )

dfd = demogs[(demogs['professional-diagnosis'] == False) & (demogs['diagnosis-year']>1900)]
print('number with no prof diag but do have diagnosis year:', dfd.shape[0] )

dfd = demogs[(demogs['professional-diagnosis'] == False) & \
             (demogs['diagnosis-year']>1900) & (demogs['onset-year']>1900)]
print('number with no prof diag but do have diagnosis year and onset year:', dfd.shape[0] )

dfd = demogs[(demogs['professional-diagnosis'] == False) & \
             (demogs['diagnosis-year'].isnull()) & (demogs['onset-year'].isnull())]
print('number with no prof diag , no diagnosis year, no onset year :', dfd.shape[0] )

nodiagorOnsethc = set(dfd.healthCode)

print('number with no prof diag, no diagnosis year, no onset year & do have a voice file :',
      len(set.intersection(nodiagorOnsethc,voiceWithHC)))

dfd.age.hist()

diagyearSurvey = len(dfd.healthCode.isin(scoreshc))

print('number with no prof diag but do have onset year who have submitted survey:', diagyearSurvey )
diagyearSurvey.age.hist()

dfd = demogs[(demogs['professional-diagnosis'] == True) & (demogs.voiceFile==True) &
             demogs.healthCode.isin(scoreshc)]
print('number with professional diag & voice file & submitted UPDRS:', dfd.shape[0] )
PwPscoresVf = set(dfd.healthCode)

dfd = scores[scores.healthCode.isin(PwPscoresVf)]
print('updrs distribution for PwPs with Voice file\n',dfd['MDS-UPDRS2.1'].value_counts())
dfd['MDS-UPDRS2.1'].value_counts()
len(set(dfd[dfd['MDS-UPDRS2.1']==4].healthCode))
PwpVoicehc = set(dfd.healthCode)



dfd = demogs[(demogs['professional-diagnosis'] == True) & \
             (demogs.voiceFile==True) & (demogs.walkFile == True)].shape
print('number with professional diag & voice file & walk file:', dfd[0] )
             
dfd = demogs[(demogs['professional-diagnosis'] == True) & \
             (demogs.voiceFile==True) & (demogs.walkFile == True)].copy()

pwpwithallhc = dfd.healthCode[dfd.healthCode.isin(scoreshc)]

print('number with professional diag & voice file & walk file & updrs score:',
     len(pwpwithallhc))
# --- medical conditions 
df3[(df2.timediff >14400000) & (df3.updrs2_1 > -1)].shape

demogs.rename(columns={'health-history':'hh'},inplace=True)

def hhFrequency(hhseries):
    print('len hhseries',len(hhseries))
    hhseries = hhseries[~hhseries.isnull()]
    hhcounts = {}
    for col in hhseries:
        for word in col.split(','):
           hhcounts[word] = hhcounts.get(word,0) + 1
    return hhcounts

# comorbidities for whole dataset  
PwPhh = hhFrequency(demogs[demogs.healthCode.isin(PwPdiagnosed)].hh)
for k in sorted(PwPhh,key=PwPhh.get, reverse=True):
    print(k,':',PwPhh[k])
    
# comorbs for sample 
PwPSamplehc = set(df.healthCode[df.diagnosed==True] )
nPwPSamplehc = set(df.healthCode[df.diagnosed!=True] )
len(PwPSamplehc)

PwPsamphh = hhFrequency(demogs[demogs.healthCode.isin(PwPSamplehc)].hh)

for k in sorted(PwPsamphh,key=PwPsamphh.get, reverse=True):
    print(k,':',PwPsamphh[k])
    
nPwPsamphh = hhFrequency(demogs[demogs.healthCode.isin(nPwPSamplehc)].hh)

for k in sorted(nPwPsamphh,key=nPwPsamphh.get, reverse=True):
    print(k,':',nPwPsamphh[k])
    
    
demogs[demogs.healthCode.isin(PwPdiagnosed)].shape

demogs.shape[0] - len(PwPdiagnosed)
for k in sorted(PwPhh,key=PwPhh.get, reverse=True):
    print(k,':',PwPhh[k])
   
        


# ---   UPDRS Score distributions ------------
# distribution of updrs scores in total ... we have a list of healthcodes that
# have one or more records in scores - need the distribution of updrs2.1, 2.12, 2.13 
# for all records and on a limited basis [why limited ? ]
pwpwithallScores   = scores[scores.healthCode.isin(PwpVoicehc)].copy()
#join age to scores 
pwp2 = pwpwithallScores.merge(demogs,on='healthCode',how='left')
len(pwpwithallScores)
Updrs21 = pwpwithallScores['MDS-UPDRS2.1'].value_counts()

Updrs210 = pwpwithallScores['MDS-UPDRS2.10'].value_counts()
Updrs212 = pwpwithallScores['MDS-UPDRS2.12'].value_counts()
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlabel('UPDRS score ')
ax.set_ylabel('Number')
ax.set_title(r'MDS-UPDRS Scores Distribution for PwPs with voice file')

dd = pd.concat([Updrs21,Updrs210,Updrs212],axis=1)
dd.sort_index().plot(kind='bar',ax=ax,zorder=3)
plt.grid(True,zorder=0)
plt.show()
#join age to pwps 
pwp2.plot.scatter(x='age',y='MDS-UPDRS2.1') # not very informative 
# for each age bin, plot updrs1 score 
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(1,1, figsize=(12, 5))

fig, ax1 = plt.subplots(1,1, figsize=(12, 5))

ax1.set_xlabel('Age')
ax1.set_ylabel('Number')
ax1.set_title(r'Age profile of PwP - UPDRS 0 ')
pp1 = pwp2[pwp2['MDS-UPDRS2.1']==0].age.hist(bins=10,ax=ax1)
#ax2.set_xlabel('Age')
#ax2.set_ylabel('Number')
#ax2.set_title(r'Age profile of PwP - UPDRS 1 ')

pwp2[pwp2['MDS-UPDRS2.1']==1].age.hist(bins=10,ax=ax1)
#ax3.set_xlabel('Age')
#ax3.set_ylabel('Number')
#ax3.set_title(r'Age profile of PwP - UPDRS 2 ')
pwp2[pwp2['MDS-UPDRS2.1']==2].age.hist(bins=10,ax=ax1)
#ax4.set_xlabel('Age')
#ax4.set_ylabel('Number')
#ax4.set_title(r'Age profile of PwP - UPDRS 3 ')
pwp2[pwp2['MDS-UPDRS2.1']==3].age.hist(bins=10,ax=ax1)
ax1.legend()
plt.show()

# need to repeat for max of 3 per person - which three ? - just exclude people with > 3 files ? 

scores['numsubs'] = scores.healthCode.apply(lambda x: \
      len(scores.healthCode[scores.healthCode==x]))

pwpwithallScoreslt4   = scores[(scores.numsubs <4) & scores.healthCode.isin(PwpVoicehc)]

Updrs21 = pwpwithallScoreslt4['MDS-UPDRS2.1'].value_counts()

Updrs210 = pwpwithallScores['MDS-UPDRS2.10'].value_counts()
Updrs212 = pwpwithallScores['MDS-UPDRS2.12'].value_counts()

dfd = demogs[(demogs['professional-diagnosis'] == True) & \
             (demogs.voiceFile==True) & demogs.inscores==True]

scores.numsubs.value_counts()
# --- Variation in UPDRS score per subject
# changes in UPDRS scores per subject - variance on PwPs with > 1 recors across 2_1, 2_10, 2_12
# minimum of 3 each ? 
#for each subject , need vector which we can measure varaance on 
 

#how many people in scores have no diagnosis, and updrs scores of 0 ?
# looking for nonpwp who have filled in score

dfcomplete.columns
dfpwp.columns
dfpwp[(dfpwp['professional-diagnosis']==False) & (dfpwp['MDS-UPDRS2.12']==0)
&(dfpwp['onset-year']>1900)
& (dfpwp['MDS-UPDRS2.10']==0)& (dfpwp['MDS-UPDRS2.1']==0)
& (dfpwp['MDS-UPDRS2.4']==0) & (dfpwp['MDS-UPDRS2.5']==0)
& (dfpwp['MDS-UPDRS2.6']==0) & (dfpwp['MDS-UPDRS2.7']==0)
& (dfpwp['MDS-UPDRS2.13']==0) 
& (dfpwp['MDS-UPDRS2.8']==0) & (dfpwp['MDS-UPDRS2.9']==0)].shape

dfpwp[(dfpwp['professional-diagnosis']==False) & 
( 
 (dfpwp['MDS-UPDRS2.12']>0) | (dfpwp['MDS-UPDRS2.10']>0) 
| (dfpwp['MDS-UPDRS2.1']>0)
| (dfpwp['MDS-UPDRS2.4']>0) | (dfpwp['MDS-UPDRS2.5']>0)
| (dfpwp['MDS-UPDRS2.6']>0) | (dfpwp['MDS-UPDRS2.7']>0)
| (dfpwp['MDS-UPDRS2.13']>0) 
| (dfpwp['MDS-UPDRS2.8']>0) | (dfpwp['MDS-UPDRS2.9']>0)
)].shape

dfpwphc = set(dfpwp.healthCode.values)
len(dfpwphc)
# for updrs2_12 classes, what is the distibution of the updrs2_10 ad 2_1 ?
scores['MDS-UPDRS2.1'].value_counts() 
for updrs in np.arange(-1,5):
    restricted = scores[scores['MDS-UPDRS2.12']==updrs]['MDS-UPDRS2.1'].value_counts()
    actual = scores['MDS-UPDRS2.1'].value_counts()  
    print('class ',updrs,'excess :', actual - restricted) 
#%%
''' get audio file for each processed walking file ''' 
#2 audio files available 
import synapse1 as syn1
from synapse1 import setGlobals, initDatasets
syn1.setGlobals()
syn1.initDatasets()
imp.reload(syn1)
syn1.scores.shape
syn1.syn1.voiceDf.shape
syn1.demogs.shape
# number with diagnosis = yes 
demogs[demogs['professional-diagnosis'].isnull()].shape
demogs.shape
# get distribution of updrs scores for those with walk and voice files  

def plotDemo(dfd):
    
    fig, ax = plt.subplots()
    a_heights, a_bins = np.histogram(dfd['MDS-UPDRS2.1'])
    b_heights, b_bins = np.histogram(dfd['MDS-UPDRS2.12'], bins=a_bins)
    c_heights, c_bins = np.histogram(dfd['MDS-UPDRS2.10'], bins=b_bins)
    width = (a_bins[1] - a_bins[0])/2
    
    ax.bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue')
    ax.bar(b_bins[:-1]+width, b_heights, width=width, facecolor='seagreen')
    ax.bar(c_bins[:-1]+2*width, c_heights, width=width, facecolor='yellow')
    plt.show()


#%% 
