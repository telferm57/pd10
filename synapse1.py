# -*- coding: utf-8 -*-#
"""
Retrieve and manipulate files from mPower Synapse repository 

"""
#%%
# pip install synapseclient


#synapse login -u me@example.com -p secret --rememberMe
   
    
#==============================================================================
# def synapseSetup(user,pw):
#     syn = synapseclient.login(email=user,password=pw, rememberMe=True)
#    
#     return syn
#==============================================================================
# run this prior : pip install (--upgrade) synapseclient[pandas,pysftp]

import synapseclient
import synapseutils
import pandas as pd
import numpy as np 
import datetime 
import configparser
import logging, imp
import cfg, pdb
import shutil
import csv
import os
imp.reload(cfg)
logging.INFO
cfg.DEMOG
#logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
logging.basicConfig(level=logging.INFO) # does not seem to work ? 
logging.getLogger().setLevel(logging.INFO)
config = configparser.ConfigParser()
config.read('./mjfoxdata/mjfox.ini')
user = config['userdetails']['user']
pw = config['userdetails']['pw'] 
#%%
def setGlobals():
    global g_set
    global g_dict
    global pwpCounter # balancing pwp and npwp 
    global nonpwpCounter
    global nonpwpWritten
    global updrsScores 
    global updrsScoresWritten
    global featureFileCounter
    global healthcodesProcessed
    global demogs
    global scores
    g_set = set()
    g_dict = {}
    pwpCounter = 0 # balancing pwp and npwp 
    nonpwpCounter =0
    nonpwpWritten =0
    updrsScores = {1:0,-1:0,0:0,2:0,3:0,4:0,5:0,None:0,'nan':0}
    updrsScoresWritten = {1:0,-1:0,0:0,2:0,3:0,4:0,5:0,None:0,'nan':0}
    featureFileCounter = 0
    healthcodesProcessed = set()
    
def initDatasets():
    ''' load demographic and updrs data  '''
    getScores(0)
    getDemographics(0)
    

def synapseSetup(user,pw):
    
    #synapse login -u me@example.com -p secret --rememberMe
    syn = synapseclient.login(email=user, password=pw, rememberMe=True)
    return syn
  


# all file names pertaining to a session now downloaded to a csv ...
# get all pedometer files 


# want to add metadata info to each file ... date , phone type, medical info 


#==============================================================================
# setofkeys= set()
# setofkeyvals= set()
#==============================================================================

def getListofSplitableCols(df): 
    # takes a dataframe, inpects each column .. 
    # if it contains all dict items , it will assemble the keys of each dict and the length of each dict into a set:setofkeys
    # returns a list of columns that are splitable (i.e. contain consistent dictionaries)
    result =[]
    for col in df.columns:
        setofkeys= set()
        #print(f2[col][1]['y'])
        nonDictItems = df[col].apply(lambda x: not isinstance(x,dict)).sum()  
        if nonDictItems ==0:
            checkKeys = df[col].apply(lambda x: setofkeys.update([item for sublist in [x.keys(), str(len(x.keys()))] for item in sublist]))
            # if we have only 1 number and an equal number of keys, we know all dicts are the same
           # print('will split dict in col ',col,'keys found are: ',setofkeys, 'LENGTH OF COL', len(df[col]) )
            logging.debug('will split dict in col %s . keys found are: %s .'
                        ' LENGTH OF COL %i', col, setofkeys, len(df[col]) )
            result.append(col)
        else:
            # assume they are just values rather than structure _ 
            logging.debug('Column ', col,' has items that are not type dict  - number of non dict items = ',nonDictItems)
            if nonDictItems != len(df): print('Warning! mixture of dict/nondict in column ',col) 
            #check they are all nondict .
    return result 
  
def splitDictCols(cols,df): 
# expects all cols to be splittable ... i,e, contain a dict with equal keys , returns df with ONLY new cols 
    rdf=pd.DataFrame()
    
    for col in cols:
        t=df[col]
      #  gg = t.apply(lambda x: pd.Series(list(x.values()),index=[x.keys()]))
        gg = t.apply(lambda x: pd.Series(x))
        # creates df from keys 
        gg.rename(columns=lambda x: col+'_'+str(x),inplace=True) 
        # check length same as df 
        assert(len(gg)==len(df))
        rdf = pd.concat([rdf,gg],axis=1)
        
    return rdf
#%%
def getMetaData(headers,fileId, colId, dfResults): # don't need headers = colnames in th df  
    """ construct dict of values of each non-filehandleid type column in the row that contains fileid in colid  """
    #print('getMetaData fileid',fileId,' colId:',colId)
    #print('type of fileId ',type(fileId))
    #TODO sort out the type checking 
    fileId = int(fileId)
    
    row = dfResults.loc[dfResults[colId]==fileId]
      
    metaData = {}
    
    for col in np.arange(1,len(headers)):
         if headers[col]['columnType']==u'FILEHANDLEID':
             continue
         name=headers[col]['name']
         if ((str(name)!= "ROW_ID") & (str(name) != "ROW_VERSION")):
             # not in dataframe .. could remove them first I guesss
             metaData[name]=row[name].values[0]
             
    return metaData 

def getDemographics(uid=0):
    ''' add demographics to file 
    read it in and return if uid=0 
    find the health code
    return the demographics '''
    global demogs
    if uid == 0:
        demogs = pd.read_csv(cfg.WRITE_PATH + cfg.DEMOG,header=0)
        return
    
    testid='03e0586a-8cf0-4e7b-a6e3-cbfecbdcee25'
    return demogs[demogs['healthCode']==uid]   
    
def getScores(uid=0,date=0):
    ''' add demographics to file 
    read it in 
    find the health code
    return the demographics '''
    global scores
    if uid==0:
        scores = pd.read_csv(cfg.WRITE_PATH + cfg.SCORES,header=0)
        return
    else:
        testid='a350041c-5c94-4b31-a608-9e64547bbfed' 
        return scores[scores['healthCode']==uid] 
    # can return more than 1 score .... return most recent for the moment 
    #TODO get the one that is closest to date passed in  
  


 #%%            
#==============================================================================
# md = getMetaData(resultsHeaders,5413083,'accel_walking_outbound.json.items',dfResults)       
# for key in md:
#     print(key,md[key])
#==============================================================================
def testFileSplit(path,fileHandleId):
    name='testing'
    dfFile = pd.read_json(path) #json file converted to df
    print('Read ', fileHandleId, 'Path ',path)
    colsToSplit=getListofSplitableCols(dfFile)
    splitColsDf =splitDictCols(colsToSplit,dfFile)
     # make up some metadata metaData = getMetaData(resultsHeaders,fileHandleId,name,dfResults) # returns all the string items as dict 
    metaData = {'add':'57 Stanley'}
    #add each metadata item to df 
    for key in metaData:
       splitColsDf[key]=metaData[key]
    dfFile.to_csv(cfg.WRITE_PATH + name + str(fileHandleId) + '.csv4') 



def buildCSVs(handles,name,resultsHeaders,dfResults,offset): 
    """ read all json files downloaded by synapse client, 
    convert json to csv, split dict columns to separate rows, add metadata 
    (the data such as date and healthcode contained in the row that contains
    the  pointer to the data file )
    build a dict of sets that contain all column variations for a file within
    the column being downloaded (name) (to check all rows have the same dictionary keys )"""
    colset=set()
    global pwpCounter # balancing pwp and npwp 
    global nonpwpCounter
    global updrsScores
    global nonpwpWritten
    global featureFileCounter
    global updrsScoresWritten
   
    
    
    for fileHandleId, path in handles.items():
    #print(file_handle_id,path)
    #with open(path) as f:
        
        dfFile = pd.read_json(path) #json file converted to df
        colconcat =''
        for c in dfFile.columns: colconcat += c + '_'
        print('colconcat ',colconcat)
        colset.add(colconcat)
        g_dict[name]=colset # building a dict of all column combinations 
        colsToSplit=getListofSplitableCols(dfFile)
        splitColsDf =splitDictCols(colsToSplit,dfFile) 
        #removeSplitCols
        dfFile.drop(colsToSplit,axis=1,inplace=True)
        dfFile = pd.concat([dfFile,splitColsDf],axis=1)
        # for each file, add all string and date  info from syanpse row that contains the file 
        metaData = getMetaData(resultsHeaders,fileHandleId,name,dfResults) # returns all the string items as dict 
        #add each metadata item to df 
        for key in metaData:
            dfFile[key]=metaData[key]
        #pdb.set_trace()
        logging.info('processing healthcode: %s : file handle : %s', metaData['healthCode'], str(fileHandleId))
        demog = getDemographics(metaData['healthCode'])
        if len(demog)==0:
            print('user not found in dem file for: ',metaData['healthCode'])
            gender = None
            age = None
            professionalDiagnosis = None
        else:
            age= demog['age'].iloc[0]
            gender=demog['gender'].iloc[0]
            professionalDiagnosis = demog['professional-diagnosis'].iloc[0]
            
        scores = getScores(metaData['healthCode'])
        
        if len(scores) > 0:
            updrs2_12 = scores.iloc[0,]['MDS-UPDRS2.12']
            if updrs2_12 > 0: pwpCounter += 1
        else:
            updrs2_12=-1
            nonpwpCounter += 1
        
        try:
            updrsScores[updrs2_12] += 1 
        except:
            logging.warning('key error :',updrs2_12)
            continue # ditch this record 
        
        logging.info('pwp %i npwp %i npwp written to file: %s',pwpCounter,nonpwpCounter,nonpwpWritten)
            
        if (nonpwpWritten - pwpCounter)>2 and (updrs2_12 in {-1,0}):
            logging.info('unbalanced pwp/non - not building files for %s',metaData['healthCode'] )
            logging.info('updrs2_12: %i',updrs2_12)        
        else:
            healthcodesProcessed.add(metaData['healthCode'])
            dfFile['gender'] = gender
            dfFile['age'] = age
            dfFile['professionalDiagnosis'] = professionalDiagnosis
            dfFile['updrs2_12'] = updrs2_12
            if name=='accel_walking_outbound.json.items': 
                print('writing features for ',metaData['healthCode'],' updrs: ',updrs2_12,'diff ',nonpwpWritten - pwpCounter)
                featureFileCounter += 1
                writeAccFeatures(dfFile,featureFileCounter,offset)
        
                if updrs2_12 in {-1,0}: nonpwpWritten +=1
                updrsScoresWritten[updrs2_12] += 1
            #dfFile.to_csv(cfg.WRITE_PATH + name + fileHandleId + '.csv2',compression='gzip') 
      
        
#%%

def getSynapseData(entity,rowLimit=0,offset=0,resultsOnly=0):
    '''Returns the sql data, not the embedded files '''    
    schema = syn.get(entity)
#schema = syn.get(table.schema.id)

    if rowLimit == 0:
        results = syn.tableQuery("select * from %s" % (schema.id))
    else:
        results = syn.tableQuery("select * from %s limit %s offset %s" % (schema.id, rowLimit, offset))
        
    if resultsOnly:
          return None, results
    else:
        dfResults = results.asDataFrame()
    return dfResults, results

def getSynapseDataSize(entity):    
    schema = syn.get(entity)
#schema = syn.get(table.schema.id)
    results = syn.tableQuery("select count(*) from %s" % (schema.id))
    return results.asInteger()
#%%
def fileDownloader(dfResults,results,colLimit=1,offset=0,col='All'): # dataframe of synapse queriy results  plus header - col types and names 
    """ for all rows passed in dfresults (generated from a synapse sql-like query),
    download the files refered to in the rows(either all or using col name
    passed), call buildCSV to split dicts within columns of 
    those dowloaded files (if there are any), add metadata as cols to the df,
    then write to csv  """
    #TODO run the query, put in df (in dfresults at the moment)
    #get the file handles 
 # headers contain a dict for each column, with 3 items:  
#                       type, id and name, except for the first 2, that has  just type and name 
    downloadCount=0
    resultsHeaders=results.headers
    header=results.headers
    items = set([
                    # 'deviceMotion_walking_outbound.json.items',
                 'accel_walking_outbound.json.items'
                 #'accel_walking_rest.json.items'
                 ])
    
    # if there are FILEHANDLEIDs in the data, each file can be processed by this method. If there aren't 
    # the file can be used directly 
    for k in np.arange(0,len(header)):
        ## down loads all files in columns of type filehandleid
        name = header[k]['name'] # column name 
        if name not in items: 
            print('name:',name,' not in - breaking') 
            continue
        print('column type ',header[k]['columnType'])
        if header[k]['columnType']==u'FILEHANDLEID': #type = filehandleid if pointing to a file (could be csv or json or jpeg....etc)
          if col =='All' or name==col:
              print('downloading ',name )
           # download entire column (synapse function)  using file handles in results - downloads is dict 
              downloadedFiles = syn.downloadTableColumns(results,[name]) 
              downloadCount += 1 
           # now I want to split and rebuild these files before moving on to next column 
           # just so I can get the output completed in an incremental way - may not be the best way
              buildCSVs(downloadedFiles,name,resultsHeaders,dfResults,offset) # name is name of col containing files 
        if downloadCount == colLimit:
           break
       
#%%

    
def getAllData(entity,maxfiles,offset=0):
    '''gets all data from synapse table - all the data files if there are any 
    (via fileDownloader) or the whole table if there are no files '''
    #schema = syn.get(table.schema.id)
    setGlobals()
    
    howbig = getSynapseDataSize(entity)
    print('records found: ',howbig)
    dfResults, results = getSynapseData(entity,maxfiles,offset)  #limit,offset
   
   
    print('shape', dfResults.shape)
    rhDf = pd.DataFrame(results.headers) 
    
    if 'FILEHANDLEID' in rhDf.columnType.values:
        # if any files in the results, down load them and split to csv's
        colLimit = 50 # effectively all columns  
        #if howbig > rowLimit: dfResults, results = getSynapseData(entity,rowLimit,offset)
        fileDownloader(dfResults,results,colLimit,offset)
    else: # no contained files - download as table  
        dfResults, results = getSynapseData(entity,maxfiles,offset)
        dfResults.to_csv(cfg.WRITE_PATH + results.tableId + '.csv5') 
        print('written',results.tableId)


#%%
#==============================================================================
# # signal processing
#==============================================================================

import synsigp as sp
#==============================================================================
# testFile = 'accel_walking_outbound.json.items5398957.csv2'
# testDf = pd.read_csv(cfg.WRITE_PATH + testFile,header=0,compression='gzip')
# testDf.head()
#==============================================================================
def writeAccFeatures(df,uid,offset):
    '''Calls synsigp. features to calculate accelerometer features, adds extra rows
    for demographics and id (to allow comparison with other data ) '''
    df['magnitude'] = sp.magnitude(df)
    addF = ['age','gender','professionalDiagnosis','updrs2_12','healthCode']
    toadd = list(df[addF].iloc[0])
    
    
    with open(cfg.WRITE_PATH + 'accel/FeaturesAcc' + str(uid+offset) +'.csv', 'w',newline='') as out:
        rows = csv.writer(out)
        for f in sp.features(df):
            rows.writerow(f + toadd)

#%%
def getSynapseRecord(entity,healthId):
    return
def getDataFile(entityDf,healthCode,col,entity):
    '''write file to path or returns -1 if not found in dataframe that contains all metadata of to be downloaded  file  
    as selected from synapse table entity '''
    print('getDataFile')
    recs = entityDf[entityDf.healthCode == healthCode]
    lr = len(recs)
    if lr == 0:
        return -1
    else:
        if len(recs)>2: logging.info('more than 2 recs, downloading first. No recs: %s' % lr)
        for i in range(len(recs)):
           # print(type(recs))
            rowIdAndV = recs.recordId.index.values[i]
            print(rowIdAndV)
            file_info = syn.downloadTableFile(entity, rowIdAndVersion=rowIdAndV,
                                               column=col,
                                               downloadLocation='.')
            break # only downloaading first record

        return file_info
    
#%%
''' main code '''
#log in 
#syn=synapseSetup(user,pw)
mjfoxFiles = { 'demographics':'syn5511429','walking':'syn5511449','updrs':'syn5511432', 'voice':'syn5511444'}

#get first set of walking files data files - bypassing getAllData function for the moment 
fetchRecords=500
walkDf , walkresults = getSynapseData(mjfoxFiles['walking'],fetchRecords,0,resultsOnly=0)
colLimit=20
offset = 0 
accelout ='accel_walking_outbound.json.items'
setGlobals()
fileDownloader(walkDf,walkresults,colLimit,offset,col=accelout)
audioDownloaded = set()

len(healthcodesProcessed)
for healthcode in healthcodesProcessed:
    print('Healthcode',healthcode)
    loc = getDataFile(voiceDf,healthcode,'audio_audio.m4a',mjfoxFiles['voice'])
    print(loc)
    if loc==-1:
        logging.info('no audio file for %s',healthcode)
    else:
        shutil.copy2(loc['path'], 'F:/DS/diss/data/audio/' + healthcode + '.m4a')
        audioDownloaded.add(healthcode)
     
    os.remove(loc['path'])
    
len(audioDownloaded)
testid='03e0586a-8cf0-4e7b-a6e3-cbfecbdcee25'

def createAudioFeatures():
    ''' once the features have been extracted using praat, the demographics
    of each participant are added, then all are written to features file '''

initDatasets()

audioFeaturesPath = 'F:\\DS\\diss\\data\\audio\\features\\'
audioDf = pd.DataFrame()
DemogToAdd = ['healthCode','packs-per-day','age','professional-diagnosis','gender']
      
for name in os.listdir(audioFeaturesPath):
    healthCode = name.split('.',1)[0]
    demog = getDemographics(uid=healthCode)
    if len(demog)==0:
        print('demographics for %s not found'%healthCode)
        continue
    toAdd = demog[DemogToAdd]
    updrsScore = getScores(uid=healthCode)
    if len(updrsScore)==0:
        print('updrs scores for %s not found - setting to -1' % healthCode)
        updrs2_1 = -1
    else:
        updrs2_1 = updrsScore['MDS-UPDRS2.1'].values[0]
    print('updrs21 score %s'% updrs2_1)
    with open(audioFeaturesPath + name,'r') as readin: 
    # read each csv , add demograhhics and updrs info 
        inrow = csv.reader(readin)
        for r in inrow:  # should only be one row
            print(r)
            with open(cfg.WRITE_PATH + 'audio/features/audioFeatures.csv',
                      'a',newline='') as out:
                outrow = csv.writer(out)
                concatRow = r + list(toAdd.iloc[0,]) + [int(updrs2_1)]
                outrow.writerow(concatRow)
            
            a2 = csv.reader(audioFeaturesPath + name,header=None)
            
            
   # assuming file name = healthcode 
concatRow = r + list(toAdd.iloc[0,]) +[-1]
int(-1)
       #a2 = pd.concat([a2,toAdd],axis=1)
       #audioDf = pd.concat([audioDf,a2],axis=0)
       
audioDf.head()
a2
# use lamexp to convert to mp3 - manual step
# use praat scipt to create feature files - manual step  
name = 'asdasd-d-d.fg'
healthCode = name.split('.',1)[0]
healthCode  
         
# get corresponding  audio files 
# process audio file 
# produce features for audio file 
#%% 
syn=synapseSetup(user,pw) # log in to synapse
import imp
imp.reload(sp)
import time
mjfoxFiles = { 'demographics':'syn5511429','walking':'syn5511449','updrs':'syn5511432', 'voice':'syn5511444'}
entity = mjfoxFiles['walking']
entity = mjfoxFiles['demographics']
entity = mjfoxFiles['updrs']
testIdVoice = '639e8a78-3631-4231-bda1-c911c1b169e5'
voiceDf , voiceresults = getSynapseData( mjfoxFiles['voice'],0,0,resultsOnly=0)
# getSynapseRecord(mjfoxFiles['voice'],testIdVoice)
voiceDf.shape
voiceDf.columns
voiceDf.recordId[:5]

loc = getDataFile(voiceDf,testIdVoice,'audio_audio.m4a',mjfoxFiles['voice'])
loc
import shutil
shutil.copy2(loc['path'], 'F:/')
os.remove(loc['path'])

len(voiceDf['healthCode'])


file_info = syn.downloadTableFile(entity, rowIdAndVersion=rowIdAndV,
                                               column=col,
                                               downloadLocation='../../data/audio')
file_info
dftt = voiceDf[voiceDf.healthCode == testIdVoice]
dftt = voiceDf.head(3)
dftt.columns
type(dftt)
dftt.recordId.index.values[1]
type(aa)
type(voiceresults)
voiceDf
voiceresults.next()
i=0
for row in voiceresults:
    i += 1
    print(row)
    if i >4: break
df1
df2

 
#%% mucking around 
def tt(*arg):
    for a in arg:
        print(a)

tt(1,2,3)


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])
df1
colconcat =''
for c in df1.columns: colconcat += c + '_' 

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                           'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']}
                         )
df2[df2.A=='A5']
df0 =pd.DataFrame()
df3 = pd.concat([df0,df1,df2],axis=1)
df2
