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
import logging, imp
import synapse1
import synapseclient
import synapseutils
import pandas as pd
import numpy as np 
import datetime 
import configparser
import cfg, pdb
import shutil
import csv
import pickle
import os
import matplotlib.pyplot as plt

# for gait feature extraction 
#imp.reload(mhealthx)
#imp.reload(mhealthx.extractors.pyGait.heel_strikes)
from mhealthx.extract import run_pyGait
from mhealthx.extractors.pyGait import project_walk_direction_preheel, walk_direction_preheel
from mhealthx.extractors.pyGait import heel_strikes
from  mhealthx.xio import read_accel_json
import synsigrest
from mhealthx.signals import compute_sample_rate
import mhealthx.signals
imp.reload(mhealthx.signals)
logging.INFO
cfg.DEMOG


logger = logging.getLogger('mthead')
fh = logging.FileHandler("synapse1_11_08_18.log",mode='w')
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
lh = logger.handlers.copy()
for handler in lh:
    logger.removeHandler(handler)
    
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(logging.INFO)
logger.info("app Start again 2")


#logger.removeHandler(logger.handlers[1])

#logger.removeHandler(logger.handlers[0])
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
    global audiofPath    
    global audioFn 
    global audioProbDiagFn
    global restPath
    global restFeatureCols
    global accelout 
    global accelrest 
    global devicerest 
    global syn, user, pw
    global mjfoxFiles
    global walkDf, walkresults
    global voiceDf, voiceresults
    global accPath


    audioProbDiagFn = 'probdiag.csv'
    audiofPath = 'audio/features/'
    g_set = set()
    g_dict = {}
    pwpCounter = 0 # balancing pwp and npwp 
    nonpwpCounter =0
    nonpwpWritten =0
    updrsScores = {1:0,-1:0,0:0,2:0,3:0,4:0,5:0,None:0,'nan':0}
    updrsScoresWritten = {1:0,-1:0,0:0,2:0,3:0,4:0,5:0,None:0,'nan':0}
    featureFileCounter = 0
    healthcodesProcessed = {}
    mjfoxFiles = {'demographics':'syn5511429','walking':'syn5511449','updrs':'syn5511432', 'voice':'syn5511444'}
    restPath = 'accel\\rest\\'
    accPath =   'accel\\walk\\'
    accelout = 'accel_walking_outbound.json.items'
    accelrest = "accel_walking_rest.json.items"
    devicerest = "deviceMotion_walking_rest.json.items"     
    mjfoxFiles = { 'demographics':'syn5511429','walking':'syn5511449','updrs':'syn5511432', 'voice':'syn5511444'}
    config = configparser.ConfigParser()
    config.read('./mjfoxdata/mjfox.ini')
    user = config['userdetails']['user']
    pw = config['userdetails']['pw'] 
        
    
    
    
    audioFn ='audioFeatures.csv'

def getDemographics(uid=0,byrecId=False):
    ''' add demographics to file 
    read it in and return if uid=0 
    find the health code
    return the demographics '''
    global demogs
    if uid == 0:
        demogs = pd.read_csv(cfg.WRITE_PATH + cfg.DEMOG,header=0)
        return
    else:
    #testid='03e0586a-8cf0-4e7b-a6e3-cbfecbdcee25'
        if byrecId:
            return demogs[demogs['recordId']==uid] 
        else:
            return demogs[demogs['healthCode']==uid]   
    
def getScores(uid=0,date=0,byrecId=False):
    ''' add demographics to file 
    read it in 
    find the health code
    return the demographics '''
    global scores
    if uid==0:
        scores = pd.read_csv(cfg.WRITE_PATH + cfg.SCORES,header=0)
        scores.dropna(inplace=True,subset=['MDS-UPDRS2.12','MDS-UPDRS2.1'])
        scores['numsubs'] = scores.healthCode.apply(lambda x: \
              len(scores.healthCode[scores.healthCode==x]))
        return
    else:
        #testid='a350041c-5c94-4b31-a608-9e64547bbfed'
        if byrecId:
            return scores[scores['recordId']==uid] 
        else:
            return scores[scores['healthCode']==uid] 
    # can return more than 1 score .... return most recent for the moment 
    #TODO get the one that is closest to date passed in
#getScores(uid='00547584-0c04-4228-a5d5-c68f7d59f176')

#vv = getScores(uid='0428d0cf-c2a4-4630-b0d7-3e4939b4784c',date=0,byrecId=True)

def getAudioUPDRSProb(uid=0):
    global audioProbs
    if uid != 0:
        print('lo')
        return audioProbs[audioProbs.healthCode==uid]
    else: 
        audioProbs = pd.read_csv(cfg.WRITE_PATH + audiofPath + audioProbDiagFn)
        
#getScores(uid='00547584-0c04-4228-a5d5-c68f7d59f176')
#getAudioUPDRSProb(uid='00547584-0c04-4228-a5d5-c68f7d59f176')
    
def initDatasets():
    ''' load demographic and updrs data  '''
    # login 
    global syn
    syn=synapseSetup(user,pw) # if not logged in 
    getScores(0)
    getDemographics(0)
    getVoiceMetaData(0)
    getWalkMetaData(0)
    # getAudioUPDRSProb(0) 
#TODO sort out filep paths for updrs probs

def getVoiceMetaData(flag):
    global voiceDf, voiceresults
    voiceDf , voiceresults = getSynapseData( mjfoxFiles['voice'],0,0,resultsOnly=0)
    

def getWalkMetaData(flag):
    global walkDf, walkresults
    walkDf , walkresults = getSynapseData( mjfoxFiles['walking'],0,0,resultsOnly=0)
 
#%%
def synapseSetup(user,pw):
    
    #synapse login -u me@example.com -p secret --rememberMe
    syn = synapseclient.login(email=user, password=pw, rememberMe=True)
    return syn


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


# all file names pertaining to a session now downloaded to a csv ...
# get all pedometer files 


# want to add metadata info to each file ... date , phone type, medical info 


#==============================================================================
# setofkeys= set()
# setofkeyvals= set()
#==============================================================================
#%%
def getListofSplitableCols(df): 
    ''' takes a dataframe, inpects each column .. 
    if it contains all dict items, it will assemble the keys of each dict
    and the length of each dict into a set:setofkeys
    returns a list of columns that are splitable (i.e. contain consistent
    dictionaries)'''
    result =[]
    for col in df.columns:
        setofkeys= set()
        #print(f2[col][1]['y'])
        nonDictItems = df[col].apply(lambda x: not isinstance(x,dict)).sum()  
        if nonDictItems ==0:
            checkKeys = df[col].apply(lambda x: setofkeys.update(
                    [item for sublist in [x.keys(), str(len(x.keys()))] for item in sublist]))
            # if we have only 1 number and an equal number of keys, we know all dicts are the same
           # print('will split dict in col ',col,'keys found are: ',setofkeys, 'LENGTH OF COL', len(df[col]) )
            logger.debug('will split dict in col %s . keys found are: %s .'
                        ' LENGTH OF COL %i', col, setofkeys, len(df[col]) )
            result.append(col)
        else:
            # assume they are just values rather than structure _ 
            logger.debug('Column %s  has items that are not type dict  - number of non dict items = %i', col,nonDictItems)
                   
            if nonDictItems != len(df): print('Warning! mixture of dict/nondict in column ',col) 
            #check they are all nondict .
    return result 
  
def splitDictCols(cols,df): 
# expects all cols to be splittable ... i,e, contain a dict with equal keys ,
# returns df with ONLY new cols 
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
    """ construct dict of values of each non-filehandleid type column in the 
    row that contains fileid in colid  """
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

#%%
def extractGait(input_file):
    # input_file = cfg.WRITE_PATH + '/testfiles/' + 'accel_walking_return-test-1.csv'
   # basePath = "C:\\Users\\dad\\.synapseCache\\428\\5389428\\428\\5389428\\"

    #tf2 = "accel_walking_outbound.json.items-502d0501-43ee-4462-a7e4-012e216576922296422479994601681.tmp"
   # input_file = basePath + tf2
    start = 150
    device_motion = False
    t, axyz, gxyz, uxyz, rxyz, sample_rate, duration = read_accel_json(input_file, start, device_motion)
    if sample_rate == 0: return -1,'File of too short duration'
    
    ax, ay, az = axyz
    len(axyz)
    type(axyz)
    stride_fraction = 1.0/8.0
    threshold0 = 0.5
    threshold = 0.2
    order = 4
    cutoff = max([1, sample_rate/10])
    distance = None
    #row = pd.Series({'a':[1], 'b':[2], 'c':[3]})
    row = pd.Series({'a':[1], 'b':[2], 'c':[3]})
    file_path = 'test1.csv'
    table_stem = cfg.WRITE_PATH + '/testfiles/features'
    save_rows = False
    #walk_direction_preheel
    #def walk_direction_preheel(ax, ay, az, t, sample_rate, 
    #                          stride_fraction=1.0/8.0, threshold=0.5,
    #                         order=4, cutoff=5, plot_test=False)
   
    

   # directions = walk_direction_preheel(ax, ay, az, t, sample_rate,\
   #                                             stride_fraction, threshold0, order, cutoff)
 
    
    try:
        px, py, pz = project_walk_direction_preheel(ax, ay, az, t, sample_rate,\
                                                stride_fraction, threshold0, order, cutoff)
        
        feature_row = run_pyGait(py, t, sample_rate,
                                            duration, threshold, order,
                                            cutoff, distance, row, file_path,
                                            table_stem, save_rows)
    except (ValueError, IndexError) as e:
        return -1,e
    else:
        return 1,feature_row

 #%%            

def testFileSplit(path,fileHandleId):
    ''' not used - delete '''
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
def writeAccFeatures(df,fileCounter,offset,gaitFeatures,gaitRow,includeAudio,
                     returnDf=False,runNo=1,addF=[],wsize=100):
    global accPath
    '''Calls synsigp. features to calculate accelerometer features, adds extra rows
    for demographics and id (to allow comparison with other data )
    data will either be written to file or returned as DF '''
    #import pdb
   # pdb.set_trace()
    df['magnitude'] = sp.magnitude(df)
    #TODO make audioprob dependent on if buildcsv has been called with audioprob = yes
    # addF = to be added to the computed features
   
    # build features dataframe 
   # pdb.set_trace()
    accf = []
    
    for f in sp.features(df,windowSize=wsize): # sp.features is a generator 
        accf.append(f)
   
    accFeatures = pd.DataFrame(accf) 
    accFeatures.columns = sp.featureList1
    if returnDf:
       return accFeatures # return pure features 
 #TODO get rid of this mess here ! Need to hand off writing the df and adding columns 
 # elsewhere 
     
    toadd = df[addF].iloc[0].values # to pass to feature extractor to add to extracted features
 
    accFeaturesMean = pd.DataFrame(accFeatures.mean()).T
    accFeaturesStd = pd.DataFrame(accFeatures.std()).T
    allAddFeatures = np.concatenate([gaitRow,toadd])
    gf = pd.DataFrame(data=allAddFeatures).T
    featuresDf = pd.concat([accFeaturesMean,gf],axis=1,ignore_index=True)
    featuresDf = featuresDf.applymap(lambda x: 0 if x==None else x )    
    featuresDf.to_csv(cfg.WRITE_PATH + accPath + str(runNo) +'\\FeaturesAcc' + str(fileCounter+offset) +'.csv',header=False,index=False)
    
#    with open(cfg.WRITE_PATH + 'accel/FeaturesAcc' + str(fileCounter+offset) +'.csv',
#              'w',newline='') as out:
#        rows = csv.writer(out)
#      
#        for f in sp.features(df): # sp.features is a generator 
#            rows.writerow(f + toadd)
#%%

def buildCSVs(handles,name,resultsHeaders,dfResults,offset,
              includeAudio=False,restrictHealthCodes=0,
              restart=False,runNo=1):
    """ read all json files downloaded by synapse client, 
    convert json to csv, split dict columns to separate rows, add metadata 
    (the data such as date and healthcode contained in the row that contains
    the  pointer to the data file )
    build a dict of sets that contain all column variations for a file within
    the column being downloaded (name)
    (to check all rows have the same dictionary keys) 
    to limit the number of subject's files in sample (some subjects can 
    have 100's of files) , set restricthealthcode to 
    limit-1 """
    
    #TODO need to remove this ugliness ! 
    colset=set()
    global pwpCounter # balancing pwp and npwp 
    global nonpwpCounter
    global updrsScores
    global nonpwpWritten
    global featureFileCounter
    global updrsScoresWritten
    global restPath
    global accPath
    global scores
   
    fileCounter = 0 # total files processed, whether selected or not 
    featureFileCounter = 0 # feature files written 
    logger.info('BuildCSV: Start of run: offset %i', offset)
    #restart = True # ignores all healthcode in healthcodesProcessed
    
        
    for fileHandleId, path in handles.items():
        
        fileCounter += 1
        print('Files Processed:',fileCounter,'Files With Features extracted:',
              featureFileCounter,'offset this run:',offset)
    #print(file_handle_id,path)
    #with open(path) as f:
       # path = 'F:\DS\diss\data\testfiles\accel_walking_return-test-1.json'# actually json 
       # path = 'F:\DS\diss\data\testfiles\accel_walking_rest.json.items5392228.csv2'
       # fn = 'accel_walking_outbound.json.items-b1d9e4f7-a13c-422e-888f-64925374cb7a2045775211023462443.tmp'
     #   path = 'F:\\DS\\diss\\data\\testfiles\\' + fn 
        if offset< 0: #indicates metadata already retrieved 
            datatime=path[1]
            medTimepoint= path[2]
            healthCode = path[3]
            path = path[0]
            this_healthCode =healthCode # dict keyed on record ID 
            this_createdOn = datatime
            
        dfFile = pd.read_json(path) #json file converted to df
        
        # for certain walk files (device motion), x,y,z contained in a dict in a json item 
        # this process splits them into seperate cols in the df 
        colconcat =''
        for c in dfFile.columns: colconcat += c + '_'
        # print('colconcat ',colconcat)
        colset.add(colconcat)
        g_dict[name]=colset # building a dict of all column combinations 
        colsToSplit=getListofSplitableCols(dfFile) # applies to device motion files 
        splitColsDf =splitDictCols(colsToSplit,dfFile) 
        #remove Cols that have been split 
        dfFile.drop(colsToSplit,axis=1,inplace=True)
        dfFile = pd.concat([dfFile,splitColsDf],axis=1)
        ##  end of xyz in a dict processing 
       # pdb.set_trace()
        if offset< 0: #indicates metadata already retrieved  
            dfFile['healthCode']=this_healthCode # added below
            dfFile['createdOn']=this_createdOn
            dfFile['medTimepoint']=medTimepoint
        
        else:    
        # for each file, add all string and date  info from syanpse row that contains the file 
            metaData = getMetaData(resultsHeaders,fileHandleId,name,dfResults) # returns all the string items as dict 
        #add each metadata item to df 
            for key in metaData:
                dfFile[key]=metaData[key]
        #pdb.set_trace()
            logger.info('processing healthcode: %s : file handle : %s', metaData['healthCode'], str(fileHandleId))
        #TODO only get demographics if you're going to write the file 
            this_healthCode = metaData['healthCode']
            this_createdOn = metaData['ceatedOn']
        # for testing : this_healthCode='008b878d-8b12-428a-99bb-d39e1db26512'
        
        if restart and (this_healthCode in healthcodesProcessed): continue
            
        if restrictHealthCodes: # limit # files per healthcode
            if this_healthCode in healthcodesProcessed:
                if healthcodesProcessed[this_healthCode] > restrictHealthCodes:
                    print('Healthcode exceeded limit files already - skipping this one',
                          this_healthCode,"limit:",restrictHealthCodes)
                    logger.info('Healthcode %s has %i files already - skipping this one',
                                this_healthCode,restrictHealthCodes)
                    continue # ditch this record 
                    
        demog = getDemographics(this_healthCode)
        if len(demog)==0:
            print('user not found in dem file for: ',this_healthCode)
            gender = None
            age = None
            professionalDiagnosis = None
        else:
            age= demog['age'].iloc[0]
            gender=demog['gender'].iloc[0]
            professionalDiagnosis = demog['professional-diagnosis'].iloc[0]
            try:
                onset_year = 2015 - demog['onset-year'].iloc[0]
            except:
                onset_year = 0 
# --- get scores            
        this_recId = fileHandleId # dict passed in contains recid as key 
        # this_createdOn = 0 
        if offset < 0: recScores = getScores(this_recId,this_createdOn,byrecId=True)
        else: recScores = getScores(this_healthCode,0,byrecId=False)
        
        motorCols = ['MDS-UPDRS2.1','MDS-UPDRS2.4','MDS-UPDRS2.5',
                         'MDS-UPDRS2.6','MDS-UPDRS2.7','MDS-UPDRS2.8',
                         'MDS-UPDRS2.9','MDS-UPDRS2.10','MDS-UPDRS2.12',
                         'MDS-UPDRS2.13']
        newCols = list(map(lambda x: x.lower()[4:].replace('.','_'), motorCols))

        if len(recScores) > 0:
            # Note we are retrieving the score record by recordid, set when 
            # downloading the datafile  
            
            
        # this_recId='b2e41b19-2b22-4c94-9a95-afcd30fd3974'        
  
            # if targetVal > 0: pwpCounter += 1
            if professionalDiagnosis: pwpCounter += 1
             # add all motor related scores to dataframe 
            motorScores = recScores.iloc[0,][motorCols] # should only be one record now using recid 
            for old, new in zip(motorCols,newCols):
                dfFile[new]=motorScores[old]
         
        else: # assume it is in demographics with no updrs scores 
         
            nonpwpCounter += 1
            # add in -1 for all motorscores 
            for old, new in zip(motorCols,newCols):
                dfFile[new]=-1

        
        logger.info('pwp %i npwp %i npwp written to file: %s',pwpCounter,nonpwpCounter,nonpwpWritten)
   
#----------   balancing now achieved by record selection           
#           this is only relevant for mass download approach 
#         
#        if (nonpwpWritten - pwpCounter)>2 and (targetVal in {-1,0}) \
#            and offset> -1: # offset = -1 if sample already balanced 
#           logger.info('unbalanced pwp/non - not building files for %s',this_healthCode )
#            logger.info('updrs2_12: %i',targetVal)        
       
            #TODO build a feature list here ! 
            
        if this_healthCode in healthcodesProcessed:
            healthcodesProcessed[this_healthCode] += 1
        else:   
            healthcodesProcessed[this_healthCode] = 1
        dfFile['gender'] = gender
        dfFile['age'] = age # big imbalance of age to < 30 whereas all pd's are 40+ 
        dfFile['professionalDiagnosis'] = professionalDiagnosis
        # dfFile[updrsMap[target]] = targetVal # all scores now in dataframe - see above       
        dfFile['healthCode'] = this_healthCode
        dfFile['onset_year'] = onset_year
        
        if includeAudio:
            # this_healthCode='008b878d-8b12-428a-99bb-d39e1db26512'
           # this_healthCode='008b878d-8eeeeeeeeeeeeb-d39e1db26512'
            audioProb = getAudioUPDRSProb(uid=this_healthCode)
            if len(audioProb)==0:
                print('buildcsv: user not found in audioprob file for: ',this_healthCode)
                audioProb = 0.5
            else:
                audioProb = audioProb['1'].values[0]
            
            dfFile['audioProb'] = audioProb
        # addf: which columns to add to features from dfFile    
        addF = ['age','gender','professionalDiagnosis','onset_year','updrs2_1','updrs2_10',
                'updrs2_12','updrs2_13','healthCode']
        if includeAudio:
            addF.insert(4,'audioProb')    
        if name=='accel_walking_outbound.json.items': 
             # get gait features 
            # path = 'F:\DS\diss\data\testfiles\accel_walking_return-test-1.csv'
       
            gaitrc, gaitFrame = extractGait(path)
            if gaitrc == -1:
                logger.warning('Extract gait failed : %s',gaitFrame)
                continue # get next record 
            else:
                gaitFeatures = gaitFrame.columns.values
                gaitFeaturesRow = gaitFrame.iloc[0].values
               
          
            # write the features out    
            print('writing features for ',this_healthCode,' updrs: ',
                  targetVal,'recID: ', this_recId,'diff ',nonpwpWritten - pwpCounter)
            featureFileCounter += 1
            writeAccFeatures(dfFile,featureFileCounter,offset,gaitFeatures,
                             gaitFeaturesRow,includeAudio,runNo=runNo,addF=addF,wsize=100)
# --- devicerest 
        if name in {accelrest,devicerest}:
            deviceFile = False
            if name == devicerest: deviceFile = True
            featureFileCounter += 1
            # feature extraction  - synsigrest uses the x,y,z info ni df to return features 
            restFeatures = synsigrest.getRestFeatures(dfFile,this_healthCode,includeAudio,deviceFile)
             # just write file out - no feature extraction yet ! 
            print(restFeatures)
            restDf = pd.DataFrame(data=restFeatures)
            accelDf= writeAccFeatures(dfFile,featureFileCounter,offset,[],
                                 [],includeAudio,returnDf=True,addF=addF,wsize=200)
            accFeaturesMean = pd.DataFrame(accelDf.mean()).T
            toadd = dfFile[addF].iloc[0].values
            toaddDf = pd.DataFrame(data=toadd).T
            toaddDF.columns=addF
            rest2Df = pd.concat([restDf,accFeaturesMean,toaddDf],axis=1)
#            rest2Df['gender'] = gender
#            rest2Df['age'] = age # big imbalance of age to < 30 whereas all pd's are 40+ 
#            rest2Df['professionalDiagnosis'] = professionalDiagnosis
#            
#            rest2Df['healthCode'] = this_healthCode
#            if includeAudio: rest2Df['audioProb'] = audioProb
             # copy path to rest directiory
            outpath = cfg.WRITE_PATH + restPath
            fileSuffix = offset + fileCounter
            rest2Df.to_csv(outpath + 'restfile_' + str(fileSuffix) + '.csv',
                          index=False, header=False)
            global restFeatureCols
            restFeatureCols = rest2Df.columns.values
            rest2Df.head(2)
            for b in restFeatureCols: print(b)
           # for n in restFeatureCols:print(n)
            #restFeatureCols = ['HR1','maxEntropy','rmsEntropy','gender','age',
            #                   'professionalDiagnosis','updrs2_10',
            #                   'healthCode','audioProb']
               
        

#%%
def fileDownloader(dfResults,results,colLimit=1,offset=0,
                   col='All',restrictHealthCodes=0): 
    # dataframe of synapse queriy results  plus header - col types and names 
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
#TODO remove items ?     
    items = set([ col
                    # 'deviceMotion_walking_outbound.json.items',
                 #'accel_walking_outbound.json.items'
                 #'accel_walking_rest.json.items'
                 ])
    
    # if there are FILEHANDLEIDs in the data, each file can be processed by this method. If there aren't 
    # the file can be used directly 
    for k in np.arange(0,len(header)):
        ## down loads all files in columns of type filehandleid
        name = header[k]['name'] # column name 
        #if name not in items: 
         #   print('name:',name,' not in list of items to download - breaking') 
        #    continue
        print('column type ',header[k]['columnType'])
        if header[k]['columnType']==u'FILEHANDLEID': #type = filehandleid if pointing to a file (could be csv or json or jpeg....etc)
          if col =='All' or name==col:
              print('fileDownLoader: downloading ',name )
           # download entire column (synapse function)  using file handles in results - downloads is dict 
           # the amount of files is determined by results of query passed in to this function 
           # and how many of the files are already in the local cache
              downloadedFiles = syn.downloadTableColumns(results,[name]) 
              downloadCount += 1 
           # now I want to split and rebuild these files before moving on to next column 
           # just so I can get the output completed in an incremental way - may not be the best way
              #TODO unhardcode target 
              buildCSVs(downloadedFiles,name,resultsHeaders,dfResults,
                        offset,True,restrictHealthCodes,target='MDS-UPDRS2.10') 
        if downloadCount == colLimit:   
           break
       
#%%

    
def getAllData(entity,maxfiles,offset=0):
    '''gets all data from synapse table - all the data files if there are any 
    (via fileDownloader) or the whole table if there are no files. As this 
    currently gets ALL the files (so 6 lots of accelerometer data, for example)
    , I am currently bypassing this '''
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
def getSynapseRecord(entity,healthId):
    return
def getDataFile(entityDf,healthCode,col,entity):
    '''write file to path or returns -1 if not found in dataframe that contains
    all metadata of to be downloaded  file  
    as selected from synapse table entity '''
    #TODO pass date in to get datafile closest ot date (which may be date from UPDRS score, for example )
    #TODO allow caller to determine ohw many records they want 
    print('getDataFile')
    recs = entityDf[entityDf.healthCode == healthCode]
    lr = len(recs)
    if lr == 0:
        return -1
    else:
        if len(recs)>2: logger.info('more than 2 recs, downloading first. No recs: %s' % lr)
        for i in range(len(recs)):
           # print(type(recs))
            rowIdAndV = recs.recordId.index.values[i]
            print(rowIdAndV)
            file_info = syn.downloadTableFile(entity, rowIdAndVersion=rowIdAndV,
                                               column=col
                                              # ,downloadLocation="."
                                              )
            break # only downloaading first record - there can be many (1-150) 

        return file_info
    
def addToAudioFeatures(outpath,outFn,voiceRun,hcFn,matchFile):
    ''' once the audio features have been extracted using 
    Praat, the demographics of each participant are added,
    then all are written to one features file. 
    format of match file: 
       key - the specific recid of the participant 
       path - path to data file 
       timestamp of data file 
       medpoint
       time difference between datafile and scores/demo record 
       healthcode 
        
    [n.b. for walking data, this is done in buildCSV, one file at a time, 
    along with feature extraction ]'''

    #initDatasets()
    fileData = pickle.load(open(hcFn,'rb'))
    audioFeaturesPath = outpath
    
    DemogToAdd = ['healthCode','packs-per-day','age','professional-diagnosis',
                  'gender','onset-year','diagnosis-year']
          
    for name in os.listdir(audioFeaturesPath):
        #   name='afa37ee2-2713-43ba-8a51-5c6fa563680e.txt' # no updrs 
        # name='afa7746f-ebd0-4311-9b9a-81f34f665fd6.txt' # with updrs 
        healthCode = name.split('.',1)[0] # healthcode is record id for match data 
        # need to get the updrs score closest in time to the time recorded 
        if matchFile:  # get healthcode from scores 
           
            recId = healthCode
            # fileDataList - will be added to feature file 
            # 1 = time and 2 - medpoint 3- timediff 
            fileDataList = [fileData[recId][i] for  i in [1,2,3]] 
            #recId could be from scores or demographics - cannot tell from id 
            this_updrs = getScores(uid=recId,date=0,byrecId=True)
            if len(this_updrs>0):
                this_updrs = this_updrs.iloc[0] #this_updrs is DF
                healthCode = this_updrs.healthCode
                scoretime = this_updrs['createdOn']
                updrs2_1 = this_updrs['MDS-UPDRS2.1']
                if np.isnan(updrs2_1): updrs2_1 = -2
                updrs2_10 = this_updrs['MDS-UPDRS2.10']
                if np.isnan(updrs2_10): updrs2_10 = -2
                updrs2_12 = this_updrs['MDS-UPDRS2.12']
                if np.isnan(updrs2_12): updrs2_12 = -2
            else: # get it from healthcodes  
            
                this_updrs = getDemographics(uid=recId,byrecId=True)
                this_updrs = this_updrs.iloc[0] 
                healthCode = this_updrs.healthCode
                updrs2_1 = -1
                updrs2_10 = -1
                updrs2_12 = -1
                scoretime = 0
        else: # old style - keyis healthcode 
            fileDataList = [fileData[healthCode][i] for i in [1,2,3]] # time and medpoint
            updrsScore = getScores(uid=healthCode)
            updrs2_1 = -1
            if len(updrsScore)==0:
                logger.warning('updrs scores for %s not found - setting to -1',healthCode)
            else: # this gives us any valid updrs score ... it will currently be the last valid UPDRS score 
                for rec in updrsScore[['MDS-UPDRS2.1','MDS-UPDRS2.10']]:
                    print(rec)
                    if np.isnan(rec):
                        continue
                    else:
                        updrs2_1 = rec
            
        demog = getDemographics(uid=healthCode) 
        if len(demog)==0:
            logger.warning('demographics for %s not found',healthCode)
            continue
        toAdd = demog[DemogToAdd]
        
            
        logger.debug('updrs21 score %s for healthCode %s',updrs2_1,healthCode)
       
        
        with open(audioFeaturesPath + name,'r') as readin: 
        # read each csv , add demograhhics and updrs info 
            inrow = csv.reader(readin)
            rcounter = 0 
            for r in inrow:  # should only be one row
                print(r)
                rcounter += 1
                if rcounter > 1: 
                    print('warning!: more than 1 row in ',name,' breaking out' )
                    continue
                with open(audioFeaturesPath + outFn,
                          'a',newline='') as out:
                    outrow = csv.writer(out)
                    concatRow = r + list(toAdd.iloc[0,]) + fileDataList + \
                     [int(updrs2_1)] + [int(updrs2_10)] + [int(updrs2_12)]
                    outrow.writerow(concatRow)
                
#TODO add info record count   
    return
                    
def convertToNumeric(x):
    ''' returns int or float if x is int, float or string version thereof
    otherwise returns np.nan'''
    
    if isinstance(x, int):
        return x
    elif isinstance(x, float):
        return x
    elif isinstance(x, str):
        if x.translate({ord('-'):'',ord('.'):''}).isnumeric():
            return float(x)
        else: return np.nan
    else: return np.nan    
                   
def cleanAudioFeatures(audiofPath, audioFn):
    #TODO unhack this ! 
    audioCompletePath = audiofPath + audioFn
    ff = pd.read_csv(audioCompletePath,header=None,na_values='--undefined--')
    colnames = ['medianPitch','meanPitch','sdPitch','minPitch',
                'maxPitch','nPulses','nPeriods','meanPeriod', 
	     'sdPeriod',
	    'pctUnvoiced',
	   # 'fracUnvoiced',
	    'nVoicebreaks', 
	    'pctVoicebreaks',
	    #'degreeVoicebreaks',
	    'jitter_loc',
	    'jitter_loc_abs',
	    'jitter_rap', 
	    'jitter_ppq5', 
	    'shimmer_loc',
	    'shimmer_loc',
	    'shimmer_apq3', 
	    'shimmer_apq5',
	    'shimmer_apq11', 
	    'mean_autocor',
	    'mean_nhr',
	    'mean_hnr',
        'healthCode','packs_per_day','age','diagnosed','gender','onset_year',
        'diagnosis_year','datetime','medPoint','timediff','updrs2_1',
        'updrs2_10','updrs2_12'
    ]
    ff.columns = colnames   
       # replace --undefined-- with mean for column 
    ff['gender']=ff['gender'].map({'Male':1,'Female':0})
    ff['packs_per_day']=ff['packs_per_day'].fillna(0)
    ff['onset_year']=ff['onset_year'].fillna(0)
    ff['diagnosis_year']=ff['diagnosis_year'].fillna(0)
    ff.medPoint = ff.medPoint.map({"I don't take Parkinson medications":0,
                "Immediately before Parkinson medication":3,
                "Just after Parkinson medication (at your best)":1,
                "Another time":2})

    from pandas.api.types import is_string_dtype
    if is_string_dtype(ff['mean_autocor']): # if string in column, numerics and missing cope --undefined -- will be strings 
        ff['mean_autocor'] = ff['mean_autocor'].apply(lambda x: x if x.translate({ord('-'):'',ord('.'):''}).isnumeric() else 0 )
    # remove non numeric - replace with NaN - save healthcodes to put back  
    #TODO get rid of this healthcodes kludge ! must be a way to apply 
    # function to subset of rows 
    ff_healthCodes = ff['healthCode']
    ff.drop('healthCode',axis=1,inplace=True) # stopping healthcode getting converted to nan 
    ff2 = ff.applymap(convertToNumeric) 
    ff['healthCode'] = ff_healthCodes 
   # ff2.isnull().sum()
    ff2['healthCode']=ff_healthCodes       
    logger.info('CleanAudioFeatures: before drop %s',len(ff2))
    
    # remove all records with undefined median pitch ~15% and with blank diagnosis (~1%)
    ff2.dropna(axis=0,subset=['medianPitch','diagnosed','gender'],inplace=True)
    ff2_healthCodes = ff2['healthCode'] #  save healthcodes again 
    ff2.drop('healthCode',axis=1,inplace=True)
  
    logger.info('CleanAudioFeatures: after drop %s',len(ff2))
    ff2.fillna({'medPoint':0},inplace=True) # nan medpoints = 0 (i.e. 'I dont take')
    ff2 = ff2.apply(lambda x: x.fillna(x.mean()),axis=0)
    ff2['healthCode']=ff2_healthCodes 
    ff2['healthCode'].isnull().sum()
    # check healthcode updrscore matches input 
#  # cannot do this with matched files - up to 2 records per health code    
#    for hc in ff2['healthCode']: 
#        ff2hc = ff2.updrs2_1[ff2.healthCode == hc] 
#        ffhc = ff.updrs2_1[ff.healthCode == hc]
#        if len(ffhc) > 1:
#            print('more than one record for %s',hc)
#            continue
#        if ffhc.iloc[0] != ff2hc.iloc[0]: print(ff2hc,ffhc)
       
    
    ff2.to_csv(audiofPath + 'cleaned' + audioFn,index=False)
 
    
    # performing the followuing manuallly in Excel : 
    #TODO - remove all those with diagnosis = yes and no updrs2_1 score  (~3%)
    
    
    return

def getVoiceData(schema,syn,healthCodes,datatime,voiceDf,col):
    # date time for health code 1425904701000
    #datetime=000
    ''' get voice mp4 files for given healthcode nearest to time datatime '''
 
    ## healthCode='639e8a78-3631-4231-bda1-c911c1b169e5'
    #healthCode = 'ce2b2605-57e8-4c20-9194-af768781454d' # has 3 files 
    #     datetime=1426285645000 # time of second file 
   
    files = {}
    print('files to be processed: ',len(healthCodes))
    i = 0

    for healthCode in healthCodes:
        i+=1
        if i%10==0:print('File: ',i)
        print('processing; ',healthCode)
        #qry = 'select * from ' + schema.id + ' where healthCode="' + healthCode + '"'
        #results = syn.tableQuery(qry)
        #rdf = results.asDataFrame()
        rdf = voiceDf[voiceDf.healthCode==healthCode]
        if len(rdf)==0: print(healthCode,' not found');continue
        rdf.iloc[0]
        if datatime: 
            dtime = datatime[healthCode]
            print('datatime for walk hc',healthCode,' : ',dtime)
        else: dtime=0
        datatimes = rdf.createdOn.values
    # find closest in time  
        idx = np.argmin(np.abs(datatimes - dtime))
        row = rdf.index[idx] # selected row (rowID and _version version nearest in time nearest in time 
        # to check if we already have this file - need file number ... 
        # synapse caching handles this 
        #filehandle = rdf.iloc[idx][col]
        path = syn.downloadTableFile(schema.id, rowIdAndVersion=row, 
                column=col)
        created = rdf.iloc[idx].createdOn
        files[healthCode] = [path,created,rdf.iloc[idx].medTimepoint]
        
        if datatime: 
            diff = datetime.timedelta(seconds=np.int((dtime-created)//1000))
            printdiff = 'diff: days: ' + str(diff.days) + 'hours: ' + \
            str(diff.seconds//3600)
        else: printdiff='No walk time passed to function'   
            
        print('selected time from voice ', rdf.iloc[idx].createdOn)
        print(printdiff) 
    return files

def getVoiceData2(schema,syn,recIds,datatime,voiceDf,col,settype):
    # date time for health code 1425904701000
    #datetime=000
    ''' get voice mp4 files for given healthcode/recID .
        choose datafile with nearest to time to score time,
        using either a) data is dict  passed with key = record ID (could be either 
        a score recorid or a healthcode record id) 
        or b) (in development) a list of healthcodes, in which case there can 
            be multiple scores records per health code and each one needs to be matched by 
            extracting the rec id and calling the first function.'''
 
    ## healthCode='639e8a78-3631-4231-bda1-c911c1b169e5'
    # healthCode = 'ce2b2605-57e8-4c20-9194-af768781454d' # has 3 files 
    #     datetime=1426285645000 # time of second file 
   
    files = {}
    print('files to be processed: ',len(recIds))
    i = 0
    if  settype==0: # timestamps passed to function with record ids 
        for recId, healthCode in recIds.items():
            i+=1
            if i%10==0:print('File: ',i)
            
            print('processing; ',healthCode)
            #qry = 'select * from ' + schema.id + ' where healthCode="' + healthCode + '"'
            #results = syn.tableQuery(qry)
            #rdf = results.asDataFrame()
            rdf = voiceDf[voiceDf.healthCode==healthCode] # get all voice files for hc 
            if len(rdf)==0: print('Voice file for ',healthCode,' not found');continue
            
            dtime = datatime[recId] # use datatime on matching walk record
            print('datatime for walk hc',healthCode,' : ',dtime)
            datatimes = rdf.createdOn.values # timestamps of voice files
            idx = np.argmin(np.abs(datatimes - dtime)) 
            row = rdf.index[idx] # selected row (rowID and _version version nearest in time nearest in time 
            # to check if we already have this file - need file number ... 
            # synapse caching handles this 
            #filehandle = rdf.iloc[idx][col]
            path = syn.downloadTableFile(schema.id, rowIdAndVersion=row, 
                    column=col)
            created = rdf.iloc[idx].createdOn
            timediff = dtime-created
            diffseconds = np.int((dtime-created)//1000)
            diff = datetime.timedelta(seconds=diffseconds)
            printdiff = 'diff: days: ' + str(diff.days) + 'hours: ' + \
            str(diff.seconds//3600)
                                 
            print('selected time from voice ', rdf.iloc[idx].createdOn)
            print(printdiff) 
                
            files[recId] = [path,created,rdf.iloc[idx].medTimepoint,timediff,healthCode]         
                
            
      
    elif settype==1: # set of healthcodes passed from generate sample - need to find closest voice file 
        for healthCode in recIds:
            scoreDf = scores[scores.healthCode==healthCode]
            demogsDf = demogs[demogs.healthCode==healthCode]
            if len(demogsDf)==0: print(healthCode,' demog data not found - skip');continue
            rdf = voiceDf[voiceDf.healthCode==healthCode] # get all voice files for hc 
            if len(rdf)==0: print('Voice file for ',healthCode,' not found');continue
            filesPerCode = 2 # limited to 2 at the moment
            #loops = 1
            if len(scoreDf) > 0: 
            # loops = min([scoreDf.shape[0],filesPerCode])
                isScore = True
                if len(scoreDf)> 1: print(healthCode, ' number records in scores: ',len(scoreDf))
            else: isScore = False  
            matched = getClosestFile(isScore,rdf,scoreDf,demogsDf)
        
            for recId,row,rdfidx,timediff in matched:
                path = syn.downloadTableFile(schema.id, rowIdAndVersion=row,
                                             column=col)
                print('adding recordId: ',recId,' healthCode: ',healthCode )
                files[recId] = [path,rdf.iloc[rdfidx].createdOn,
                      rdf.iloc[rdfidx].medTimepoint,timediff,healthCode]  
    elif settype==2: # set of recordIds passed from generate sample - need to find closest voice file 
        for recId in recIds:
      
            scoreDf = scores[scores.recordId ==recId] # always be 1 or 0 records 
            if len(scoreDf) > 0: 
                isScore = True
                healthCode=scoreDf.healthCode.values[0]
                if len(scoreDf)> 1: print(healthCode, ' number records in scores: ',len(scoreDf))
            else: 
                isScore = False  
                demogsDf = demogs[demogs.recordId==recId] # should always be 1
                if len(demogsDf)==0: print(recId,' demog nor score data not found - skip');continue
                healthCode=demogsDf.healthCode.values[0]
                
            rdf = voiceDf[voiceDf.healthCode==healthCode] # get all voice files for hc 
            if len(rdf)==0: print('Voice file for ',healthCode,' not found');continue
         
            
            matched = getClosestFile(isScore,rdf,scoreDf,demogsDf)
            # in this case (settype = 2) the match should return one file 
            if len(matched)==0: print('Error: no voice files returned for recID ',recId)
            for recId,row,rdfidx,timediff in matched:
                path = syn.downloadTableFile(schema.id, rowIdAndVersion=row,
                                             column=col)
                print('adding recordId: ',recId,' healthCode: ',healthCode )
                files[recId] = [path,rdf.iloc[rdfidx].createdOn,
                      rdf.iloc[rdfidx].medTimepoint,timediff,healthCode]  
                            
    return files

def sinceDiag(df):
    df['onset_year']= df['onset_year'].apply(lambda x: 2015-x if x>1990 else 0)
    return 

def getHCGroups(df):
    '''returns amn array indicating the groups of healthcodes in 
    df that can be used in kGroupFolds'''
    hc = df.healthCode
    hcs = set(hc) 
    # give each healthcode a number
    dd = {k:v for k,v in zip(hcs,np.arange(len(hcs)))}  
    #
    dfx = df.copy()
    # add code in to df 
    dfx.hcgroup = dfx.healthCode.apply(lambda x: dd[x] )
    groups = dfx.hcgroup.values
#    from sklearn.model_selection import GroupKFold
#    X  = df.iloc[:,:-4].values
#    y = df.updrs2_1.values
#    len(X)
#    gkf = GroupKFold(n_splits=3)
#    groups = df.hcgroup.values
#    for train, test in gkf.split(X, y, groups=groups):
#        print("%s %s" % (train, test))
#    
#    # are the 2's split ? 
#    codes2 = df.healthCode[df.duplicated(subset='healthCode')]
#    # no hc in  codes2 should have records in different train and test groups 
#    # for each code in code2 , check only present in one of test or train  
#    for hc1 in codes2.values:
#        len(hc1)    
#        ix = df.index[df.healthCode == hc1] # all index values  for this hc 
#        for i in ix:
#            # just want to make sure they are all in test or none are in test 
#            print(i, np.where(test==i)    )
    return groups

def addmedPoint():
    return

   
def addVoiceUPDRS(df): 
    ''' add updrs2_12 scores to  voice dataframe - forgot to add it !   '''  
    dfx = df.copy() 
    dfx.reset_index(inplace=True)
    dfx['updrs2_12'] = -1
    
    for i in dfx.index: 
      
        hc = dfx.iloc[i].healthCode
        #getScores(uid=0)
        scs = getScores(uid=hc,date=0,byrecId=False).copy()
        scs.reset_index(inplace=True)
        if len(scs)==0: continue # leave it at -1 
        if len(scs)==1: dfx.loc[i,'updrs2_12'] = scs.loc[0,'MDS-UPDRS2.12']
        
        else: # get closest time
             #series of createdOn times from scores 
             diffs = abs(scs.createdOn - dfx.loc[i,'datetime'])
             idx = np.argmin(diffs)
             dfx.loc[i,'updrs2_12'] = scs.loc[idx,'MDS-UPDRS2.12']
             scs.shape
        df['updrs2_12'] = dfx.updrs2_12  
        df.updrs2_12[1]
        df['updrs2_12'].value_counts()
        
        
def getClosestFile(isScore,rdf,scoreDf,demogsDf): 
    ''' from datafile records (rdf), score records(scoreDf) and 
    demog records (demogsDf) for a particular hc , return the pairs of 
    records closest to each other in time  '''
    #get walk files 
    #pdb.set_trace()
    wt = rdf.createdOn.values
    # if record in score table, may be multiple recs pre healthcode to match 
    # to multiple data files. if only in healthcode, only one record, one timestamp 
    # use the same process to match one or more records to data files 
    if isScore: st = scoreDf.createdOn.values
    else: st = demogsDf.createdOn.values
    type(st)
    # need to match datatimes on walk files to scores/demogs createdon date 
     # st times from scores or demogs, wt times from walk files 
     # create 2d array  (abs(st-wt)) for each combo 
    wt = wt[:,np.newaxis]
    a = abs(wt-st)
    # what if there are more healthcodes than files ?       
    # just do 2 for now - scale up later - maybe easier to create new table 
    # of healthcodes matched with closest files 
  
    # case : number of data records >= number score records - each score 
    # can be matched against closest data file 
    matchedRecords = []
    if a.shape[0] > a.shape[1]:
        # for each row, closest score has index scodemix[row] 
        rdfidxs = np.argmin(a,axis=0) # ix of rdf minimum values for each score record
        
        for ix in np.arange(min(min(a.shape),5)):
            scodemoidx = ix
            rdfidx = rdfidxs[ix] 
            print('score file no.',ix,'rdf ix', rdfidx, 'min val ',a[rdfidx,ix])
            if isScore: recId = scoreDf.iloc[scodemoidx].recordId
            else: recId = demogsDf.iloc[scodemoidx].recordId
            row = rdf.index[rdfidx] # selected row (rowID and _version) for data file
            timediff = a[rdfidx,ix] # diff between score/demog timestamp and datafile timestamp
            matchedRecords.append([recId,row,rdfidx,timediff])
    # case : number of data records < number score records - each data record 
    # can be matched against closest score file 
    else:  #  cols >= rows       
        scodemidxs = np.argmin(a,axis=1) # ix of minimum values for each data record
        for ix in np.arange(min(min(a.shape),5)):
            scodemoidx = scodemidxs[ix]
            rdfidx = ix
            print('score file no.',ix,'rdf ix', rdfidx, 'min val ',a[rdfidx,ix])
            if isScore: recId = scoreDf.iloc[scodemoidx].recordId
            else: recId = demogsDf.iloc[scodemoidx].recordId
            row = rdf.index[rdfidx] # selected row (rowID and _version) for data file
            timediff = a[ix,scodemoidx] # diff between score/demog timestamp and datafile timestamp
            matchedRecords.append([recId,row,rdfidx,timediff])
                
    return matchedRecords 
            
#        
#        minv = [10e12,10e12]
#        mindx = [0,0]
#        for i in np.arange(a.shape[0]): # find 2 smallest file/score difference 
#            for j in np.arange(a.shape[1]): 
#                if a[i,j] < minv[0] : mindx[0] = [i,j]; minv[0] = a[i,j]
#                #print('a',minv,'b',mindx)
#                if (minv[0] <= a[i,j] < minv[1]) & \
#                    ([i,j]!= mindx[0]):
#                    mindx[1] = [i,j]; minv[1] = a[i,j]
#                #print('a2',minv,'b2',mindx)
#        # note for 1 score file with many data files, the top 2 closest data files will match the 
#        # one sco/demo file so mindx will contain [[closest,0],[2nd,0]] where 0 is first and only score/demo record 
#        matchedRecords = []
#        for i in np.arange(min(min(a.shape),2)): #min of a.shape is lowest of scores vs files
#         
#            # get recordid of each score/demogs record
#            #edge case: less data files than score records - # datafiles will be matched 
#            #edge case: more data files than score records - # score records will be matched
#          
#            rdfidx = mindx[i][0]
#            scodemoidx = mindx[i][1]
#            
#            if isScore: recId = scoreDf.iloc[scodemoidx].recordId
#            else: recId = demogsDf.iloc[scodemoidx].recordId
#            row = rdf.index[rdfidx] # selected row (rowID and _version) for data file
#            timediff = minv[i] # diff between score/demog timestamp and datafile timestamp
#            matchedRecords.append([recId,row,rdfidx,timediff])
            
       
     
def getWalkData2(schema,syn,healthCodes,datatime,walkDf,col,filesPerCode):

    ''' get walk files for given healthcode, add closest updrs scores to time of file 
    creation '''
 
    ## healthCode='639e8a78-3631-4231-bda1-c911c1b169e5'
    #healthCode = 'ce2b2605-57e8-4c20-9194-af768781454d' # has 3 files 
    #     datetime=1426285645000 # time of second file 
    #healthCode='23c1a565-1761-4fe4-90a6-dcc9b9a4ee47' # (4 entries in scores, 361 walk files  )

    testHC = ['faeab8f9-5e6b-48ef-8e87-3cc7e6da3e5f', ## 119 datafiles, 2 hc records 
                'faf76a8e-045c-427f-951d-8e1918f195a8',
                'fc56e64d-800c-4805-8919-c4b3b8c3d59b',
                'fd1016e9-b57b-4545-992c-f2cb9d747ebf',
                'fe2cf802-6d04-4322-9996-dfc2399f6a42',
                '51faeedb-bc76-4fe8-9d6e-5bf43b484a94'] # 7 scorefiles 
     #scores.healthCode.value_counts()
      
    files = {}
    print('No. healthcodes to be processed: ',len(healthCodes))
    fileCounter = 0
    for healthCode in healthCodes:
       # healthCode='303e809c-703c-4224-9ddf-97155a764f56'
        #  healthCode='7fb561a3-8f90-49d2-a39b-55ea3f78a17b' 1 data file, 2 scores 
        # healthCode='61f3a7b8-354a-4093-afdb-ebeacdf2ac2d' # in demogs, not scores 
        # healthCode='23c1a565-1761-4fe4-90a6-dcc9b9a4ee47' # 4 in scores , 361 files 
       
        healthCode = testHC[5]
        
        if fileCounter%10==0:print('File Count : ',fileCounter)
        print('processing; ',healthCode)
        # get walk records for health code 
        rdf = walkDf[walkDf.healthCode==healthCode]
        rdf = voiceDf[voiceDf.healthCode==healthCode]  # change
        if len(rdf)==0: print(healthCode,' data file not found - skip');continue
        # get scores records - if > 1 , create multiple entries up to filesPerCode
        scoreDf = scores[scores.healthCode==healthCode]
        demogsDf = demogs[demogs.healthCode==healthCode]
        if len(demogsDf)==0: print(healthCode,' demog data not found - skip');continue
      
        filesPerCode = 2 # limited to 2 at the moment
        #loops = 1
        if len(scoreDf) > 0: 
            # loops = min([scoreDf.shape[0],filesPerCode])
            isScore = True
            if len(scoreDf)> 1: print(healthCode, ' number records in scores: ',len(scoreDf))
        else: isScore = False  
        
        matched = getClosestFile(isScore,rdf,scoreDf,demogsDf)
        
        for recId,row,rdfidx,timediff in matched:
            
            fileCounter += 1
            path = syn.downloadTableFile(schema.id, rowIdAndVersion=row,
                                         column=col)
            print('adding recordId: ',recId,' healthCode: ',healthCode )
            files[recId] = [path,rdf.iloc[rdfidx].createdOn,
                  rdf.iloc[rdfidx].medTimepoint,timediff,healthCode]  
            

            
 
    return files    

    
#%%
''' main code '''
#log in 
def getWalkData(recordCount,offset,actType,restrictHealthCodes=0,runSuite='A',
                runSuiteSub=1):
    ''' builds filepath info for dowloaded files. downloaded en masse  '''
    runInfo ={1:[{'offset':offset,'recordCount':recordCount}]}
    setGlobals()
    initDatasets()
    global healthcodesProcessed
    # pull in previous healthcodes, 
    if runSuiteSub > 1: # pull in previous healchcodes processed 
        co = runSuiteSub
        hcp = []
        for co in range(1,runSuiteSub):
            dumpFn = 'run_' + str(runSuite) + '_' + str(co) \
                + '_healthcodesProcessed'
            print(dumpFn)        
            hcp.append(pickle.load(open(dumpFn,'rb')))
            
        healthcodesProcessed = { k: v for d in hcp for k, v in d.items() }    
        
                
        #healthcodesProcessed = pickle.load(open(,'r'))
        print('total after load of healthcodes from previous run: ',len(healthcodesProcessed))
    fetchRecords=recordCount
    offset = offset 
    walkDf , walkresults = getSynapseData(mjfoxFiles['walking'],fetchRecords,
                                          offset,resultsOnly=0)
    #walkDf.shape
    colLimit=20
    fileDownloader(walkDf,walkresults,colLimit,offset,actType[0],
                   restrictHealthCodes)
    dumpFn = 'run_' + str(runSuite) + '_' + str(runSuiteSub) \
            + '_healthcodesProcessed'
    pickle.dump(healthcodesProcessed,open(dumpFn,'wb'))
    print('total healthcodes after run ',runSuiteSub,len(healthcodesProcessed))
 
#%%    get walk data 
def setFileAvailabilityInd(walkDf,voiceDf):
    ''' add 2 fields indicating whether voice and walk files are available for the healthcode '''
    global demogs
    global scores
    walkDfHc = set(walkDf.healthCode)
    voiceDfHc = set(voiceDf.healthCode)
    scores['walkFile'] = scores.healthCode.isin(walkDfHc)
    scores['voiceFile'] = scores.healthCode.isin(voiceDfHc)
    demogs['walkFile'] = demogs.healthCode.isin(walkDfHc)
    demogs['voiceFile'] = demogs.healthCode.isin(voiceDfHc)
    # note - may be better to add file ref of closest file to scores entry 

def walkMain(): 
    syn=synapseSetup(user,pw) # if not logged in 
    mjfoxFiles = { 'demographics':'syn5511429','walking':'syn5511449','updrs':'syn5511432', 'voice':'syn5511444'}

#get first set of walking files data files - bypassing getAllData function for the moment 
#TODO rewrite this so that it loops through  x000 records at a time
 
    global accelout
    global accelrest 
    global devicerest 

    actType = {'rest':[accelrest,'MDS-UPDRS2.10'],'walkout':[accelout,'MDS-UPDRS2.12'],
               'restDevice':[devicerest,'MDS-UPDRS2.10']}
    
    runSuite = 'B' # tag each series of runs 
    runSuiteSub = 1 # tag each component  - must be int


    getWalkData(3000,12000,actType['rest'],restrictHealthCodes=2,runSuite='E',runSuiteSub=6)    

#%%

def restMain2(schema,syn,recIds,datatime,walkDf,col,runNo,device=True):
    ''' get rest accelerometer files based on sample of walk files downloaded. 
    As the rest and walk files are recorded at the same time, there should be a
    1-1 match. 
    
    1. load record IDs and healthcodes and timepoints from pickle 
    2. for each recid :
        find associated restfile from synapse 
        get scores data if any , if none set updrs = -1 
    
    3. build df to use to call buildcsv 
    
    buildCSVs(files,col,resultsHeaders,dfResults,
               offset,includeAudio=False,restrictHealthCodes=0,
               target='MDS-UPDRS2.12',restart=False,runNo=runNo) 
    '''
    global walkresults
    entity = mjfoxFiles['walking']
    schema = syn.get(entity)
    col = devicerest
    walkFiles= pickle.load(open('walk_run_' + col + '_' + str(runNo),'rb'))
     # for each entry, get rest file  for healthcode at timepoint
    restFiles = {} 
    
    for k,v in walkFiles.items():
        hc = v[3]
        datatime = v[1]
        medPoint = v[2]
         # recId = k 
         #hc = '5f5479ce-c7a4-454e-9a43-23348b215422'
        # datatime='1426016039000'
        rdf = walkDf[(walkDf.healthCode==hc) & (walkDf.createdOn==datatime)] # get all voice files for hc 
        if len(rdf)==0: print('accel  file for ',healthCode,' not found');continue
        else:
            row = rdf.index[0] #  (rowID and _version) of the only row in rdf 
            print('downloading ',row)
            path = syn.downloadTableFile(schema.id, rowIdAndVersion=row, 
                column=col)
            restFiles[k] = [path,rdf.createdOn.values[0],rdf.medTimepoint.values[0],hc]
            
    pickle.dump(restFiles,open('rest_run_' + col + '_' + str(runNo),'wb'))  
    resultsHeader ='' # not used for this call 
    dfResults = pd.DataFrame() # not used 
    offset = -1 # indicate we are passing in synapse metadata     
    # target no longer used- all updrs scores of interest used         
    buildCSVs(restFiles,col,resultsHeaders,dfResults,
               offset,includeAudio=False,restrictHealthCodes=0,
               restart=False,runNo=runNo)     
         
         
     
     


#%%

def walkMain2(runNo,restart=False):
    ''' using the approach of selecting a representative sample from the 
    demographics and updrs scores data '''
    runNo=1
    global mjfoxFiles
    syn=synapseSetup(user,pw) # if not logged in 
    
    if restart:
        pass # prevents counts and record of healthcodes processed being reset 
    else:
        setGlobals()
        initDatasets()
    
    walkDf , walkresults = getSynapseData( mjfoxFiles['walking'],0,0,resultsOnly=0)
    # construct list of healthcodes to get based on balanced updrs 0-3 (only one 4 ! )
    # plus age matached non-scored individuals 
    #step 1 - pull 100 0 100 1 100 2 89 3 from scores 
    scores.columns
  
    samplepop = generateSample(150,'MDS-UPDRS2.12') # get healthcodes 
    len(samplepop)
    # check sample has worked 
    walkDf['inSample'] = walkDf.healthCode.isin(samplepop)
    scores['inSample'] = scores.healthCode.isin(samplepop)
    
    entity = mjfoxFiles['walking']
    schema = syn.get(entity)
    col = accelout # define which datafile you want 
    files = getWalkData2(schema,syn,samplepop,0,walkDf,col,filesPerCode=2)
    # note 'files' contains score or health record id now as first item 
    
    pickle.dump(files,open('walk_run_' + col + '_' + str(runNo),'wb'))
    
    # pass these over to buildcsv to create csvs 
    # buildcsv expects this : 
    #    (handles,name,resultsHeaders,dfResults,offset,
    #          includeAudio=False,restrictHealthCodes=0,target='MDS-UPDRS2.12')
    # buildCSV gets the synapse metadata - but using this method we have it 
    # already . resultsHeaders is static for each type of activity: 
    resultsHeaders = walkresults.headers
    # offset is irrelevant in this context
    offset = -1 # negative offset is a switch for buildCsvs to use the supplied
    # dataset 

    dfResults = pd.DataFrame() # not used with this technique of geting files
    len(files)
    buildCSVs(files,col,resultsHeaders,dfResults,
               offset,includeAudio=False,restrictHealthCodes=0,
               target='MDS-UPDRS2.12',restart=False,runNo=runNo) 
    # how do we verify ? 
#%%   test area  
#walkMain2(runNo=1,restart=False) 

#tt=pickle.load(open("walk_run_accel_walking_outbound.json.items_1",'rb'))
#tt['b9150e23-66e4-41b9-bf8b-1600e33777c2']    
#xc,cx = extractGait('c:/users/dad/.synapsecache/981/5503981/981/5503981/accel_walking_outbound.json.items-e3f659bc-bf8f-4ae1-9421-0285b826e60e6719770723082287976.tmp')
def testSample():
    sample = generateSample(300,'MDS-UPDRS2.12')
    # sample should be n records per class as far as possible , plus age-matched draw from 
    sampleSize = len(sample)
    type(sample)
    scorest = scores.copy()
    scorest['insample'] = scorest.healthCode.isin(sample)
    scorest['MDS-UPDRS2.12'][scorest.insample==True].value_counts()
    scorest['healthCode'][scorest.insample==True].nunique() # unique healthcode values in scores
    # get healthcodes with voice file, healthcodes with walk file 
    # check number of demogs records = number of scores records 
    # check dist of ages in both 
    # check dist of scores 
 #%% get audio data  

def getCutScores(num,col):
        ''' get num random records for each distinct value in col '''
        #TODO add logic to ensure healthcode in voice and walking dataframe
        #num = 150;
        # age mat
        #col = 'MDS-UPDRS2.1'
        # only select UPDRS Scores submitted by pwps - diag = true 
        scomerged = scores.merge(demogs,on='healthCode',how='left',suffixes = ('ll','rr')) # get ages 
        scomerged.rename(columns={'voiceFilell':'voiceFile','recordIdll':'recordId',
                                  'walkFilell':'walkFile',},
                         inplace=True)
        PwPhc = set(demogs[(demogs['professional-diagnosis'] ==True)].healthCode)
        len(PwPhc)  
        PwPscores = scomerged[scomerged.healthCode.isin(PwPhc)]      
        len(PwPscores) 
        PwPscores.columns  
              
        vals = PwPscores[col].unique()
        dfresult = pd.DataFrame()
        scoWithBothFiles = PwPscores[(PwPscores.walkFile==True) & (PwPscores.voiceFile==True)] # remove those that have no file
        scoWithFiles = PwPscores[(PwPscores.voiceFile==True)] # select those with voice file
        scheme = 2
        if scheme == 1:
            for val in vals:
                val = 1.0
                scocut = scoWithFiles[PwPscores[col]==val]
                scocut = scocut.take(np.random.permutation(len(scocut))[:num]) 
                dfresult = dfresult.append(scocut)
        else:     
            #until we have sample size, pick from each, limiting to age range maxima
            # randomise each se tof updrs re 
            ageranges=[0,20,30,40,50,60,65,70,80,100]
            #tt = np.digitize(18,ageranges)
            #limits of nPwPs with voice files e.g.in the 8th bin there are only 3 nPwPs 
            agelimits = {1:30,2:40,3:200,4:200,5:213,6:65,7:51,8:39,9:3}
           
            scocut = {}
            agetotals = {}
            updrstotals = {}
            for val in vals:
                # build random lists of index values for each updrs 
                scoarr =  scoWithFiles[PwPscores[col]==val].index.values
                np.random.shuffle(scoarr)
                scocut[val] = list(scoarr)
            # for each value of updrs in turn, while have not reached num 
            # pop the next index number, check 
            # age limit not exceed, if not add to list 
             
            counter=[5,5,5,5,5] # set to zero when updrs exhausted 
            scoWithFiles.age.max()
            while True: # pop frm randomlists of records for each UPDRS 
                if sum(counter)==0: break
                for val in vals:
                    try:
                        idxval = scocut[val].pop()
                    except IndexError:
                        counter[int(val)] = 0 
                        continue ## ugly but hey - 
                    this_rec = scoWithFiles.loc[idxval]
                    agebin = np.digitize(this_rec.age,ageranges)
                    agebin = int(agebin)
                    if agebin > 9: 
                        print('invalid age',this_rec.age)
                        continue
                    
                    agetotals[agebin] = agetotals.setdefault(agebin,0) +1
                    
                    if agetotals[agebin] > agelimits[agebin]:
                        print('shout ive exceeded agebin ',agebin,
                              ' limit of ',agelimits[agebin], ' - ignoring this record ')
                    else: 
                        updrstotals[val]= updrstotals.setdefault(val,0)+1
                        if updrstotals[val] > num: 
                            print('limit ',num,' reached for updrs ',val)
                            counter[int(val)]=0
                        else:    
                            print('updrs total for ',val,':',updrstotals[val])
                            dfresult = dfresult.append(this_rec)
                                  
       # df.shape    
        print('value Counts for PwP: ',dfresult[col].value_counts())
        samplehc = set(dfresult.healthCode.values)
        len(samplehc)
        demogs[demogs.healthCode.isin(samplehc)].age.hist()
         # verify all sample is PD = True 
        
        demogs[(demogs.healthCode.isin(samplehc)) & \
               (demogs['professional-diagnosis']!=True)].shape
        
       #  dfresult.drop_duplicates(subset='healthCode',inplace=True)
        print('no. healthcodes: ',dfresult.shape[0])
        return dfresult



def getCutDemog(scoresDf):
    ''' get sample of healthcodes from demographic file that are not in 
    scoresfile and do NOT have an onset year (approx 475 participants have not 
    got professional diagnosis = true, but have an onset year and have submitted
    scores, so may or may not  have PD) 
    
    input: scoresDF - dataframe of records chosen for the PwP sample by getCutScores
    this is used for the age range 
    
    '''
# get age bins for records selected by getCutScores
   # scoresDf = dfresult.copy()
 # get ages - no need - now passed over from the UPDRS sample  
    #dfmerged = scoresDf.merge(demogs,on='healthCode',how='left',suffixes = ('ll','rr')) # get ages 
    count, bins = np.histogram(scoresDf.age.values,bins=[1,20,30,40,50,60,65,70,80,100],
                               range=(1,100))
# pull random ages = number in each bin ? 
#for bins get count 
#get healthcodes nPwP - no diagnosis, no onset year, no year if diag 
#    scoreshc = set(scores.healthCode)
#    demogs['scoresind']=demogs['healthCode'].isin(scoreshc)
#    demogs.columns
    demogPop = pd.DataFrame()

    for agecat in np.arange(len(bins)):
        if agecat == 0: continue
        gg = demogs[(demogs['professional-diagnosis']!=True) &
                               (demogs.voiceFile==True) &
                               (demogs.age>bins[agecat-1]) & 
                               (demogs['onset-year'].isnull()) & 
                               (demogs.age <= bins[agecat])
                               ]
        voiceandwalk = False # checking if walk file exists too - or not 
        if voiceandwalk:
            gg = gg['healthCode','recordId'][(demogs.walkFile==True)]
            
        gg = gg.reset_index()
        gg = gg.take(np.random.permutation(len(gg))[:count[agecat-1]])
        print('max age in this bin: ',bins[agecat])
        print('number required for this bin: ',count[agecat-1])
        print('number extracted for this bin that meet conditions: ',len(gg))
        demogPop = demogPop.append(gg)  
        
        #demogPop.drop_duplicates(subset='healthCode',inplace=True)
    return demogPop


# check sample demogs['insample'] = demogs.healthCode.isin(set(demogPop.healthCode))
    


#get files 
def generateSample(num,col):
    ''' generate set of healthcodes for samples taken from scores and 
    healthcodes in demog but not in scores ..... num samples are generated for 
    each value in col (a updrs score)  from scores table  then age matched on samples 
    from demographics table '''
    num = 100;
    col = 'MDS-UPDRS2.1'
    from numpy.random import RandomState
    np.random.seed(12345)
    setFileAvailabilityInd(walkDf,voiceDf) # add file availability indicators to score and demog
    scoresPop = getCutScores(num,col)
    print(scoresPop.shape)
    demogPop = getCutDemog(scoresPop)
    print(demogPop.shape)
    demogPop.age.hist()
    #sample = set.union(set(demogPop.healthCode),set(scoresPop.healthCode))
    sample = set.union(set(demogPop.recordId),set(scoresPop.recordId))
    return sample

# (note, duplicates dropped by setting )
#%%
def getVoiceStats(matchfile):
    #get matchfile with recids - the match file is the list of record IDs 
    # of health codes within the sample 
    # create dict with datatime extracted from dictionary 
    newl = {k:v[1] for (k,v) in matchRun.items()}
    len(newl)
    # join scores based on record ID 
    scores['datatime'] = scores['recordId'].apply(lambda x: newl.get(x,-1))
    len(scores[scores.datatime!=-1])
    # diff times - createdOn and dataTime
    scores['timediff'] = scores.apply(lambda x: abs(x.createdOn- x.datatime),axis=1)
    minidf=scores[scores['datatime']!=-1]
    minidf=minidf.copy()
    len(minidf)
    minidf.timediff =  minidf.timediff/60000
    minidf.timediff.hist()
    diffar = minidf.timediff.values
    diffar =diffar/60000
    #now get medpoints for walkfiles and
    runNo=1
    # using walkfiles cos i only happen to have a walk matchfile 
    #walkfiles = pickle.load(open('walk_run_' + accelout + '_' + str(runNo),'rb'))
    #ret =  np.histogram(diffar,bins=[0,30,60,120,180,240,300,360,1440,2880,])
    return

#%%    
def voiceMain(matchFile=False,audioFile=1):
    ''' downloads files (either by sampling the scores df or using a set of 
    healthcodes passed in as matchFile, then, after the manual process of 
    converting to  mp3 and  extracting features using Praat, the files are 
    embellished with demographic info and medpoint  and time by addToAudioFeatures
    and basic cleanup is done by cleanAudioFeatures '''
    
    testIdVoice = '639e8a78-3631-4231-bda1-c911c1b169e5'
    
    setGlobals()
    initDatasets()
    
    # construct list of healthcodes to get based on balanced updrs 0-3 (only one 4 ! )
    # plus age matached non-scored individuals 
    #step 1 - pull 100 0 100 1 100 2 89 3 from scores 
    scores.columns
    voiceRun = 5
    walkRun = '1'
    matchFile=False

    if matchFile:
        # pick up the record IDs of the subjects generated by the walk file selection process 
        # and use those to find voice file with nearest timestamp 
        matchRun = pickle.load(open('walk_run_accel_walking_outbound.json.items_'+walkRun,'rb'))
        voiceSample = set(matchRun.keys()) # key is now recordid - need healthcode in last field
        
        # healthCodes = { k:v[-1] for (k,v) in zip(matchRun.keys(),matchRun.values())}
        healthCodes = { k:v[-1] for (k,v) in matchRun.items()}
        
        datatimes = { k:v[1] for (k,v) in matchRun.items() } # datatime is key 1
        voiceSample = healthCodes.copy()
        settype= 0 # set of record Id's from walk file downloads 
    else:
        voiceSample = generateSample(150,'MDS-UPDRS2.1')
        voiceSample = sample
        #  note logic in getvoicedata2 will match the audio file/ updrs score times
        # check age profiles worked 
        settype = 2 # set of record IDs 
        
        datatimes = 0
        
        
        
    len(voiceSample)
    type(voiceSample)    
    
    audioFiles=['audio_countdown.m4a','audio_audio.m4a']
    
    requestedAudioFile  = 'audio_audio.m4a' # audio_countdown is not useful
    
    entity = mjfoxFiles['voice']
    schema = syn.get(entity)
    col = "audio_audio.m4a"
    #col = 'audio_countdown.m4a'
    files = getVoiceData2(schema,syn,voiceSample,datatimes,voiceDf,col,settype)
    hcFn  = 'voice_run_' + col + '_' + str(voiceRun) # save healthcodes, paths etc
    pickle.dump(files,open('voice_run_' + col + '_' + str(voiceRun),'wb'))
    
   # files2 = pickle.load(open('voice_run_2','rb'))
    
    #move files from cache to feature generation location     
  
    audioPath = cfg.WRITE_PATH + '\\audio\\' 
 
    requestedAudioFileRoot = requestedAudioFile.split('.')[0]
    audioPath = cfg.WRITE_PATH + 'audio' + '\\' + requestedAudioFileRoot + \
            '\\' + str(voiceRun) + '\\'
    #TODO MANUALLY CREATE DIRECTORY voiceRun        
    for k, v in files.items():
        shutil.copy2(v[0], audioPath + k +  '.m4a') # note for match files k 
                                                # is now recid NOT healthcode 
    ''' Convert to mp3 (manual at the moment but could automate using ffmpeg),
        extract features using praat script 
         then add in demog and score and file date & medpoint data to produce final file ''' 
    audiofPath = audioPath + 'features\\'
    audioFn ='audioFeatures.csv'
    audioCompletePath = audioPath + audiofPath + audioFn
    matchFile = True # always the case - now sample of healthcodes also generates
                    # a match file (matching datafile to demog/scores entry)
    addToAudioFeatures(audiofPath, audioFn,voiceRun,hcFn,matchFile)
    cleanAudioFeatures(audiofPath, audioFn)    

