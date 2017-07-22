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

import synapseclient
import synapseutils
import pandas as pd
import numpy as np 
import datetime 
import configparser
import cfg, pdb
import shutil
import csv
import os
# for gait feature extraction 
#imp.reload(mhealthx)
#imp.reload(mhealthx.extractors.pyGait.heel_strikes)
from mhealthx.extract import run_pyGait
from mhealthx.extractors.pyGait import project_walk_direction_preheel, walk_direction_preheel
from mhealthx.extractors.pyGait import heel_strikes
from  mhealthx.xio import read_accel_json
imp.reload(cfg)
logging.INFO
cfg.DEMOG


logger = logging.getLogger('mthead')
fh = logging.FileHandler("synapse1_19_07_18.log",mode='w')
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')


fh.setFormatter(formatter)
logger.addHandler(fh)
logger.setLevel(logging.INFO)
logger.info("app Start again 2")
logger.handlers[0].close()
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
    mjfoxFiles = { 'demographics':'syn5511429','walking':'syn5511449','updrs':'syn5511432', 'voice':'syn5511444'}

        
    
    
    
    
    
    audioFn ='audioFeatures.csv'

def getDemographics(uid=0):
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
        return demogs[demogs['healthCode']==uid]   
    
def getScores(uid=0,date=0):
    ''' add demographics to file 
    read it in 
    find the health code
    return the demographics '''
    global scores
    if uid==0:
        scores = pd.read_csv(cfg.WRITE_PATH + cfg.SCORES,header=0)
        scores.dropna(inplace=True,subset=['MDS-UPDRS2.12','MDS-UPDRS2.1'])
        return
    else:
        #testid='a350041c-5c94-4b31-a608-9e64547bbfed' 
        return scores[scores['healthCode']==uid] 
    # can return more than 1 score .... return most recent for the moment 
    #TODO get the one that is closest to date passed in
#getScores(uid='00547584-0c04-4228-a5d5-c68f7d59f176')

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
    getScores(0)
    getDemographics(0)
    getAudioUPDRSProb(0)

  

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
def writeAccFeatures(df,fileCounter,offset,gaitFeatures,gaitRow,includeAudio):
    '''Calls synsigp. features to calculate accelerometer features, adds extra rows
    for demographics and id (to allow comparison with other data ) '''
    #import pdb
   # pdb.set_trace()
    df['magnitude'] = sp.magnitude(df)
    #TODO make audioprob dependent on if buildcsv has been called with audioprob = yes
    # addF = to be added to the computed features
    if includeAudio:
        addF = ['age','gender','audioProb','professionalDiagnosis','updrs2_12','healthCode']
    else: 
        addF = ['age','gender','professionalDiagnosis','updrs2_12','healthCode']
        
    toadd = df[addF].iloc[0].values # to pass to feature extrator to add to extracted features
 
    # build features dataframe 
   
    accf = []
    for f in sp.features(df): # sp.features is a generator 
        accf.append(f)
   
    accFeatures = pd.DataFrame(accf)   
    accFeaturesMean = pd.DataFrame(accFeatures.mean()).T
    accFeaturesStd = pd.DataFrame(accFeatures.std()).T
    allAddFeatures = np.concatenate([gaitRow,toadd])
    gf = pd.DataFrame(data=allAddFeatures).T
    featuresDf = pd.concat([accFeaturesMean,gf],axis=1,ignore_index=True)
    featuresDf = featuresDf.applymap(lambda x: 0 if x==None else x )
    
    featuresDf.to_csv(cfg.WRITE_PATH + 'accel/FeaturesAcc' + str(fileCounter+offset) +'.csv',header=False,index=False)
    
#    with open(cfg.WRITE_PATH + 'accel/FeaturesAcc' + str(fileCounter+offset) +'.csv',
#              'w',newline='') as out:
#        rows = csv.writer(out)
#      
#        for f in sp.features(df): # sp.features is a generator 
#            rows.writerow(f + toadd)
#%%

def buildCSVs(handles,name,resultsHeaders,dfResults,offset,includeAudio=False):
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
   
    fileCounter = 0 # total files processed, whether selected or not 
    logger.info('BuildCSV: Start of run: offset', offset)
    
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
        dfFile = pd.read_json(path) #json file converted to df
        
        # for certain walk files, x,y,z contained in a dict in a json item 
        # this process splits them into seperate cols in the df 
        colconcat =''
        for c in dfFile.columns: colconcat += c + '_'
        # print('colconcat ',colconcat)
        colset.add(colconcat)
        g_dict[name]=colset # building a dict of all column combinations 
        colsToSplit=getListofSplitableCols(dfFile)
        splitColsDf =splitDictCols(colsToSplit,dfFile) 
        #removeSplitCols
        dfFile.drop(colsToSplit,axis=1,inplace=True)
        dfFile = pd.concat([dfFile,splitColsDf],axis=1)
        ##  end of xyz in a dict processing 
        
        # for each file, add all string and date  info from syanpse row that contains the file 
        metaData = getMetaData(resultsHeaders,fileHandleId,name,dfResults) # returns all the string items as dict 
        #add each metadata item to df 
        for key in metaData:
            dfFile[key]=metaData[key]
        #pdb.set_trace()
        logger.info('processing healthcode: %s : file handle : %s', metaData['healthCode'], str(fileHandleId))
        #TODO only get demographics if you're going to write the file 
        this_healthCode = metaData['healthCode']
        # for testing : this_healthCode='008b878d-8b12-428a-99bb-d39e1db26512'
        if this_healthCode in healthcodesProcessed:
            if healthcodesProcessed[this_healthCode] > 1:
                print('Healthcode has 2 files already - skipping this one',this_healthCode)
                logger.info('Healthcode %s has 2 files already - skipping this one',
                            this_healthCode)
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
            
        scores = getScores(this_healthCode)
        
        if len(scores) > 0:
            updrs2_12 = scores.iloc[0,]['MDS-UPDRS2.12']
            if updrs2_12 > 0: pwpCounter += 1
        else:
            updrs2_12=-1
            nonpwpCounter += 1
        
        try:
            updrsScores[updrs2_12] += 1 
        except:
            logger.warning('key error :',updrs2_12)
            continue # ditch this record 
        
        logger.info('pwp %i npwp %i npwp written to file: %s',pwpCounter,nonpwpCounter,nonpwpWritten)
            
        if (nonpwpWritten - pwpCounter)>2 and (updrs2_12 in {-1,0}):
            logger.info('unbalanced pwp/non - not building files for %s',this_healthCode )
            logger.info('updrs2_12: %i',updrs2_12)        
        else:
            #TODO build a feature list here ! 
            if this_healthCode in healthcodesProcessed:
                healthcodesProcessed[this_healthCode] += 1
            else:   
                healthcodesProcessed[this_healthCode] = 1
            dfFile['gender'] = gender
            dfFile['age'] = age
            dfFile['professionalDiagnosis'] = professionalDiagnosis
            dfFile['updrs2_12'] = updrs2_12
            dfFile['healthCode'] = this_healthCode
            if includeAudio:
                # this_healthCode='008b878d-8b12-428a-99bb-d39e1db26512'
               # this_healthCode='008b878d-8eeeeeeeeeeeeb-d39e1db26512'
                audioProb = getAudioUPDRSProb(uid=this_healthCode)
                if len(audioProb)==0:
                    print('buildcsv: user not found in audioprob file for: ',this_healthCode)
                    audioProb = None
                else:
                    audioProb = audioProb['1'].values[0]
                
                dfFile['audioProb'] = audioProb
          # get gait features 
                # path = 'F:\DS\diss\data\testfiles\accel_walking_return-test-1.csv'
            gaitrc, gaitFrame = extractGait(path)
            if gaitrc == -1:
                logger.warning('Extractgait filed : %s',gaitFrame)
                continue # get next record 
            else:
                gaitFeatures = gaitFrame.columns.values
                gaitFeaturesRow = gaitFrame.iloc[0].values
               
            # restricting to outbound json items at the moment - tmi  
            if name=='accel_walking_outbound.json.items': # write the features out
                print('writing features for ',metaData['healthCode'],' updrs: ',updrs2_12,'diff ',nonpwpWritten - pwpCounter)
                featureFileCounter += 1
                writeAccFeatures(dfFile,featureFileCounter,offset,gaitFeatures,gaitFeaturesRow,includeAudio)
        
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
            print('name:',name,' not in list of items to download - breaking') 
            continue
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
              buildCSVs(downloadedFiles,name,resultsHeaders,dfResults,
                        offset,includeAudio=True) # name is name of col containing files 
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
    
def addToAudioFeatures(outpath,outFn):
    ''' once the audio features have been extracted using 
    Praat, the demographics of each participant are added,
    then all are written to one features file for walking, this is done in 
    buildCSV'''

    initDatasets()

    audioFeaturesPath =  cfg.WRITE_PATH + outpath
    DemogToAdd = ['healthCode','packs-per-day','age','professional-diagnosis','gender']
          
    for name in os.listdir(audioFeaturesPath):
        healthCode = name.split('.',1)[0]
        demog = getDemographics(uid=healthCode)
        if len(demog)==0:
            logger.warning('demographics for %s not found',healthCode)
            continue
        toAdd = demog[DemogToAdd]
        updrsScore = getScores(uid=healthCode)
        updrs2_1 = -1
        if len(updrsScore)==0:
            logger.warning('updrs scores for %s not found - setting to -1',healthCode)
        else: # this gives us any valid updrs score ... it will currently be the last valid UPDRS score 
            for rec in updrsScore['MDS-UPDRS2.1']: 
                print(rec)
                if np.isnan(rec):
                    continue
                else:
                    updrs2_1 = rec
            
        logger.debug('updrs21 score %s for healthCode %s',updrs2_1,healthCode)
        with open(audioFeaturesPath + name,'r') as readin: 
        # read each csv , add demograhhics and updrs info 
            inrow = csv.reader(readin)
            for r in inrow:  # should only be one row
                print(r)
                with open(audioFeaturesPath + outFn,
                          'a',newline='') as out:
                    outrow = csv.writer(out)
                    concatRow = r + list(toAdd.iloc[0,]) + [int(updrs2_1)]
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
    audioCompletePath = cfg.WRITE_PATH + audiofPath + audioFn
    ff = pd.read_csv(audioCompletePath,header=None)
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
        'healthCode','packs_per_day','age','diagnosed','gender','updrs2_1'
    ]
    ff.columns = colnames   
       # replace --undefined-- with mean for column 
    ff['gender']=ff['gender'].map({'Male':1,'Female':0})
    ff['packs_per_day']=ff['packs_per_day'].fillna(0)
    
    
    ff['mean_autocor'] = ff['mean_autocor'].apply(lambda x: x if x.translate({ord('-'):'',ord('.'):''}).isnumeric() else 0 )
    # remove non numeric - replace with NaN - save healthcodes to put back  
    #TODO get rid of this healthcodes kludge ! must be a way to apply 
    # function to subset of rows 
    ff_healthCodes = ff['healthCode']
    ff.drop('healthCode',axis=1,inplace=True) # stopping healthcode getting converted to nan 
    ff2 = ff.applymap(convertToNumeric) 
    ff['healthCode'] = ff_healthcodes 
   # ff2.isnull().sum()
    ff2['healthCode']=ff_healthCodes       
    logger.info('CleanAudioFeatures: before drop %s',len(ff2))
    
    # remove all records with undefined median pitch ~15% and with blank diagnosis (~1%)
    ff2.dropna(axis=0,subset=['medianPitch','diagnosed','gender'],inplace=True)
    ff2_healthCodes = ff2['healthCode'] #  save healthcodes again 
    ff2.drop('healthCode',axis=1,inplace=True)
  
    logger.info('CleanAudioFeatures: after drop %s',len(ff2))
    
    ff2 = ff2.apply(lambda x: x.fillna(x.mean()),axis=0)
    ff2['healthCode']=ff2_healthCodes 
    ff2['healthCode'].isnull().sum()
    # check healthcode updrscore matches input 
    for hc in ff2['healthCode']:
        ff2hc = ff2.updrs2_1[ff2.healthCode == hc] 
        ffhc = ff.updrs2_1[ff.healthCode == hc]
        if len(ffhc) > 1:
            print('more than one record for %s',hc)
            continue
        if ffhc.iloc[0] != ff2hc.iloc[0]: print(ff2hc,ffhc)
       
    
    ff2.to_csv(cfg.WRITE_PATH + audiofPath + 'cleaned' + audioFn,index=False)
 
    
    # performing the followuing manuallly in Excel : 
    #TODO - remove all those with disgnosis = yes and no updrs2_1 score  (~3%)
    
    
    return


#%%
''' main code '''
#log in 
def getWalkData(recordCount,offset):
    ''' builds filepath info for dowloaded files '''
    setGlobals()
    initDatasets()
    fetchRecords=recordCount
    offset = offset 
    walkDf , walkresults = getSynapseData(mjfoxFiles['walking'],fetchRecords,offset,resultsOnly=0)
    #walkDf.shape
    colLimit=20
    accelout ='accel_walking_outbound.json.items'
    fileDownloader(walkDf,walkresults,colLimit,offset,col=accelout)
    
syn=synapseSetup(user,pw) # if not logged in 
mjfoxFiles = { 'demographics':'syn5511429','walking':'syn5511449','updrs':'syn5511432', 'voice':'syn5511444'}

#get first set of walking files data files - bypassing getAllData function for the moment 
#TODO rewrite this so that it loops through  x000 records at a time


getWalkData(8000,16000)    
 
testIdVoice = '639e8a78-3631-4231-bda1-c911c1b169e5'

voiceDf , voiceresults = getSynapseData( mjfoxFiles['voice'],0,0,resultsOnly=0)
audioDownloaded = set()

len(healthcodesProcessed)

''' get audio file for each processed walking file ''' 
#2 audio files available 
audioFiles=['audio_countdown.m4a','audio_audio.m4a']
audioPath = cfg.WRITE_PATH + '/audio/' 
requestedAudioFile  = audioFiles[1]
requestedAudioFileRoot = requestedAudioFile.split('.')[0]
audioPath = cfg.WRITE_PATH + '/audio/' + '/' + requestedAudioFileRoot +'/'
for healthcode in (healthcodesProcessed):
    if os.path.exists(audioPath + healthcode + '.m4a'):
        logger.debug('file for healthcode %s already downloaded',healthcode)
    else:
        logger.debug('Healthcode: %s ',healthcode)
        loc = getDataFile(voiceDf,healthcode,requestedAudioFile,mjfoxFiles['voice'])
        print(loc)
        if loc==-1:
            logger.info('no audio file for %s',healthcode)
        else:
            shutil.copy2(loc, audioPath + healthcode + '.m4a')
            audioDownloaded.add(healthcode)
            os.remove(loc)

#        hc1 = '0bf685c2-7adf-4818-99a9-f9df71b6a0d2'
#        hc2 = 'd29b51d3-2997-43b8-969c-3b9fc81dc703'
    
#     loc = getDataFile(voiceDf,hc1,audioFiles[0],mjfoxFiles['voice'])
#loc     
''' Convert to mp3 (manual at the moment but could automate using ffmpeg),
    extract features using praat script 
     then add in demog and score data to produce final file ''' 

audiofPath = 'audio/features/'
audioFn ='audioFeatures.csv'
audioCompletePath = cfg.WRITE_PATH + audiofPath + audioFn
addToAudioFeatures(audiofPath, audioFn)
cleanAudioFeatures(audiofPath, audioFn)
#packs per day - nan - col 26 (must add headers! -  should be 0 


            
            

#%% 
audioCompletePath = cfg.WRITE_PATH + audiofPath + audioFn
ff = pd.read_csv(audioCompletePath,header=None)

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
        'healthCode','packs_per_day','age','diagnosed','gender','updrs2_1'
    ]

ff.columns = colnames
ff['gender']=ff['gender'].map({'Male':1,'Female':0})
ff['packs_per_day'].fillna(0, inplace=True)
ff['mean_autocor'] = ff['mean_autocor'].apply(lambda x: x if x.translate(
        {ord('-'):'',ord('.'):''}).isnumeric() else 0 )

    
        

#ff2 = ff.applymap(lambda x: float(x) if x.translate({ord('-'):'',ord('.'):''}).isnumeric() else np.nan)

ff2 = ff.applymap(convertToNumeric) 

ff2.isnull().sum()
len(ff2)
ff2.dropna(axis=0,subset=['medianPitch'],inplace=True)
len(ff2)

ff2.sum()
ff2.mean()
ff3 = ff2.apply(lambda x: x.fillna(x.mean()),axis=0)
3078/600
healthCode = '523283b6-acc9-4375-ab7a-d7240cd690c3' 
updrsScore = getScores(uid=healthCode)

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
