# -*- coding: utf-8 -*-#
"""
Spyder Editor

This is a temporary script file.
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
import synapseclient
import synapseutils
import pandas as pd
import numpy as np 
import datetime 
def synapseSetup(user,pw):
    
    #synapse login -u me@example.com -p secret --rememberMe
    syn = synapseclient.login(email=user, password=pw, rememberMe=True)
    return syn
  
syn=synapseSetup(user,pw)    

 
schema = syn.get("syn5713119")
#schema = syn.get(table.schema.id)



results = syn.tableQuery("select * from %s" % schema.id)

dfResults = results.asDataFrame()
dfResults.head(10)
resultsHeaders = results.headers # headers contain a dict for each column, with 3 items:  
#                       type, id and name, except for the first 2, that has  just type and name 

# all file names pertaining to a session now downloaded to a csv ...
# get all pedometer files 


# want to add metadata info to each file ... date , phone type, medical info 

datetime.datetime.fromtimestamp(int(1426007198000/1000.0)) ## checking timestamp in correct format - it is 

# for each column , download all files, for each file tag with the date and userid etc 
# 
file1 = syn.downloadTableColumns(results,['pedometer_walking_outbound.json.items'])

WRITE_PATH = 'F:\\DS\\diss\\data\\'
setofkeys= set()
setofkeyvals= set()

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
            print ('will split dict in col ',col,'keys found are: ',setofkeys, 'LENGTH OF COL', len(f2[col]) )
            result.append(col)
        else:
            print('Column ', col,' has items that are not type dict  - number of non dict items = ',nonDictItems)
    return result 
    
def splitDictCols(cols,df): 
# expects all cols to be splittable ... i,e, contain a dict with equal keys , returns df with ONLY new cols 
    rdf=pd.DataFrame()
    for col in cols:
        t=df[col]
        gg = t.apply(lambda x: pd.Series(x.values(),index=x.keys()) ) # creates df from keys 
        gg.rename(columns=lambda x: col+'_'+str(x),inplace=True) 
        # check length same as df 
        assert(len(gg)==len(df))
        rdf = pd.concat([rdf,gg],axis=1)
        
    return rdf 
    
def getMetaData(headers,fileId, colId, df): # don't need headers = colnames in th df  
    """ constuct dict of values of each non-filehandleid type column in the row that contains fileid in colid  """
    row = df[dfResults[colId]==fileId]
    metaData = {}
    for col in np.arange(1,len(headers)):
         if headers[col]['columnType']==u'FILEHANDLEID':
             continue
         name=headers[col]['name']
         if ((str(name)<>"ROW_ID") & (str(name) <> "ROW_VERSION")): # not in dataframe .. could remove them first I guesss
             metaData[name]=row[name].values[0]
    return metaData 

             
#==============================================================================
# md = getMetaData(resultsHeaders,5413083,'accel_walking_outbound.json.items',dfResults)       
# for key in md:
#     print(key,md[key])
#==============================================================================
    
    
    
    
    

def buildCSVs(handles,name,resultsHeaders,df): 
    """ read all json files downloaded by synapse client, 
    convert json to csv, split dict columns to separate rows, add metadata   """
    for fileHandleId, path in handles.items():
    #print(file_handle_id,path)
    #with open(path) as f:
        dfFile = pd.read_json(path) #json file converted to df
        print('Read ', fileHandleId, 'Name ',name)
        colsToSplit=getListofSplitableCols(dfFile)
        splitColsDf =splitDictCols(colsToSplit,dfFile)
        
        # for each file, add date info from dfresults
        metaData = getMetaData(resultsHeaders,fileHandleId,name,df) # returns all the string items as dict 
        #add each metadata item to df 
        for key in metaData:
            dfFile[key]=metaData[key]
        dfFile.to_csv(WRITE_PATH + name + fileHandleId + '.csv4') 
        
        
        




results = syn.tableQuery("select * from %s" % schema.id) #
dfresults = results.asDataFrame()
dfresults.head(10) 



tt[1]['name']
len(tt[2])
len(tt)

for k in np.arange(0,len(tt)): ## down loads all files in columns of type filehandleid
   print(len(tt[k]))
   if tt[k]['columnType']==u'FILEHANDLEID': #type = filehandleid if pointing to a file (could be csv or json or jpeg....etc)
       name = tt[k]['name']
       print('downloading ',name )
       
       file1 = syn.downloadTableColumns(results,[name]) # download entire column (synapse function)  using file handles in results 
       writeCSVs(file1,name)
       
       
       
testfilePath = 'C:\\Users\\dad\\.synapseCache\\249\\5413249\\249\\5413249\\deviceMotion_walking_rest.json.items-fe77c42d-76c2-48d0-b923-3a0e41537a002601173981915643771.tmp'

# f2 is just one of the splittable files we are using as a sample ... after this, we need to, for each file,
# split all 'dict' columns (normally x,y,z values)  union these to the df , add the metadata as a column (phone type,created on HEALTHCODE, ETC  ) 
# and rewrite as a csv 


f2 = pd.read_json(testfilePath)
# now we have for each row, a column containing a dict (of x, y z say, that I want in colx, coly, colz )
f2.iloc[1,:]
f2.columns

    

# split each column into additional columns colname= colname_dictkey

dfresults.iloc[0,]
# get the columns to split
toSplit= getListofSplitableCols(f2)
toSplit        
splitDf = splitDictCols(toSplit,f2)

# need to add metadata as rows - this will be all columns that  are in the row that the file is in ... note all the files get downloaded at once 
# may be a bit messy as we have to find all fthe files, read then in again and add the columns ... better to do it as we download it  
# 1. get handles for all items in 1 col. (they are downloaded to tmp at this stage - the handle includes the disk location )
# 2. for each file:
#     load it ,
#     get the splitable columns,  
#     create split df 
#     get col values of row the file was in (this is in the results of the
 # query to get all the rows cintauining the datafile nui the first place )
# before writing to disk, 
#
def main(dfResults,header): # dataframe of synapse queriy results  plus header - col types and names 
    """ for all rows passed in dfresults, download files, split dicts within columns (if there are any), add metadata to cols,
    then write to csv  """
    #TODO run the query, put in df (in dfresults at the moment)
    #get the file handles 
    downloadCount=0
    for k in np.arange(0,len(header)):
        ## down loads all files in columns of type filehandleid
        print(header[k]['columnType'])
        if header[k]['columnType']==u'FILEHANDLEID': #type = filehandleid if pointing to a file (could be csv or json or jpeg....etc)
           name = header[k]['name'] # column name 
           print('downloading ',name )
           # download entire column (synapse function)  using file handles in results - downloads is dict 
           downloadedFiles = syn.downloadTableColumns(results,[name]) 
           downloadCount += 1 
           # now I want to split and rebuild these files before moving on to next column 
           # just so I can get the output completed in an incremental way - may not be the best way
           buildCSVs(downloadedFiles,name,resultsHeaders,dfResults) # name is name of col containing files 
        if downloadCount == 1:
           break
                   
main(dfResults,resultsHeaders)
       
       

        


 
#%% mucking around 
tt = dfResults[dfResults['accel_walking_outbound.json.items']==5413083]
tt.values

for key in file1:
    print (key, file1[key])

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                        'B': ['B0', 'B1', 'B2', 'B3'],
                        'C': ['C0', 'C1', 'C2', 'C3'],
                        'D': ['D0', 'D1', 'D2', 'D3']},
                        index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                           'B': ['B4', 'B5', 'B6', 'B7'],
                        'C': ['C4', 'C5', 'C6', 'C7'],
                        'D': ['D4', 'D5', 'D6', 'D7']}
                         )
df2[df2.A=='A5']
df0 =pd.DataFrame()
df3 = pd.concat([df0,df1,df2],axis=1)
df2
setofkeys.update([[1,2],3])
type(gg)
type(f2['gravity'])
t['x']

set2 = set()
d1 = {'a':1,'b':2} 
d2 = {'b':22,'a':21,'c':24} 

ds1 = pd.Series(data=[d1.values()],names=d1.keys())
ds2 = pd.DataFrame(data=[d2.values()],columns=d2.keys())

ddf = pd.concat([ds1,ds2])
ddf
l =[d2.keys(), str(3)]
    
[item for sublist in [d2.keys(),[['x','y'],['w']], str(len(d2.keys()))] for item in sublist]
len(d2.keys())
d1.keys()==d2.keys()
tt1 =['a','b']
tt2 =['b','a']
set(tt1)==set(tt2)
import json
import csv

results_list = [[1,2,3], [1,2,4]]
results_union = set().union(*[[1,2,3], [1,2,4]])
print results_union
#*[[1,2,3], [1,2,4]]

