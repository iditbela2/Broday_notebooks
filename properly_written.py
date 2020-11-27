from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly as py
import plotly.graph_objs as go
import ipywidgets as widgets
from tqdm.auto import tqdm 

import seaborn as sns

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error #=mean error (Simon 2012)
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from scipy.optimize import minimize

from scipy.spatial import distance
from sklearn.metrics.pairwise import nan_euclidean_distances

from matplotlib.lines import Line2D
from matplotlib import rcParams, cycler
from sklearn.preprocessing import PowerTransformer

import sys
from impyute.imputation.cs import fast_knn
from impyute.imputation.cs import mice

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.filterwarnings("ignore")

# Functions

def initialize(threshold, start_year):
#     PM25 = pd.read_pickle("/Users/iditbela/Documents/Broday/sent_from_Yuval/Mon_DataBase/PM25_2019") #MY COMPUTER
    PM25 = pd.read_pickle("~/Documents/PM25_2019") #ubi computer
#     PM25 = pd.read_pickle("/Users/iditbela/Documents/Broday/saved_data_from_notebooks/PM25") #MY COMPUTER
#     PM25 = pd.read_pickle("~/Documents/saved_data_from_notebooks/PM25") #Lab's computer
#     PM25 = pd.read_pickle("~/Documents/PM25_2019") #Lab's computer
    times = pd.date_range(start='2000-01-01 00:00:00', end='2019-12-31 23:30:00', freq='30Min') #one less because the last is always nan

    print(times.shape[0] == PM25.shape[0])
    
    start_idx = np.argwhere(times == start_year+'-01-01 00:00:00')[0][0]

    # reduced PM25 
    r_PM25 = PM25[start_idx:] 
    
    # which stations pass the threshold 
    idx = r_PM25.notnull().sum(axis = 0)/r_PM25.shape[0]>threshold
    r_PM25 = r_PM25.loc[:, idx]

    r_PM25.reset_index(inplace=True)
    r_PM25.drop(labels = 'index',axis=1, inplace=True)
    
    #reduced times
    r_times=times[start_idx:]

    
    return r_times, r_PM25

# create a np_array np_y_missing, fill the np_y_missing with values(nans?) where the validation
# indexes are chosen. 
# randomly choose chunks of certain size
def get_validation_index(PM25, interval_length, s, how_many, total_exclude):    
    
    # find all valid indexes
    pot_idx = np.argwhere((PM25.iloc[:,s].notnull().rolling(interval_length,min_periods=0).sum()>(interval_length-interval_length/100)).values).ravel()
#     print(np.shape(pot_idx),s)
    # in case the last index is at the end
    if (pot_idx[-1] == PM25.shape[0]-1):
        pot_idx=pot_idx[:-1]
        
    chosen_idx = pot_idx[(~np.isnan(PM25.iloc[pot_idx-interval_length,s].values)) & (~np.isnan(PM25.iloc[pot_idx+1,s].values)) & (pot_idx>=interval_length)]
    # exclude places I already took for validation of that station
    # make sure chosen_idx is non in these places or 
    chosen_idx=[x for x in chosen_idx if ((not total_exclude[x]) and (not total_exclude[x-interval_length]))]

    # randomly choose it
    chosen = np.random.choice(chosen_idx,how_many,replace=False)+1

    return (chosen-interval_length,chosen)

# dummy_df = pd.DataFrame(data = r_PM25.iloc[24011:24017,[0,1,2,5,6,21,23,24,25]].values, columns=r_PM25.iloc[24011:24017,[0,1,2,5,6,21,23,24,25]].columns,index = times[24011:24017])
# r_PM25.iloc[24011:24017,[0,1,2,5,6,21,23,24,25]]
# times[24011:24017]

# dummy_df

# a function that returns X_missing, y_missing (splits the data according to Interval Length(IL))
def return_X_y(PM25,IL):
 
    X_missing = PM25.copy()
    y_missing = PM25.copy() 

    np_r_PM25_y_mask = PM25.copy().values
    np_r_PM25_y_mask[:] = 0
    
    for s in tqdm(range(np_r_PM25_y_mask.shape[1])):
        
        total_exclude = np.zeros(np_r_PM25_y_mask.shape[0])
        for j in range(len(IL)):
        
            interval_length = int(IL[j]*2)
            how_many = int(720/IL[j])
            A,B = get_validation_index(PM25, interval_length, s, how_many, total_exclude)
            for a,b in zip(A,B):
                total_exclude[a:b] = 1
                np_r_PM25_y_mask[a:b,s] = -(j+1)  

    # ITERATE HERE AGAIN FOR j TIMES, TO GET FOR EACH
    # COLUMN j GROUPS OF DESIRED INTERVAL LENGTH FOR EVALUATION.
    # (maybe -1 to -j...)        
                      
    # y_missing 
    y_missing.iloc[:] = np.nan
    np_y_missing = y_missing.values
    np_X_missing = X_missing.values

    # put in np_y_missing 
    np_X_missing_copy = np_X_missing.copy()
    
    for j in range(len(IL)):
        np_y_missing[np_r_PM25_y_mask==-(j+1)]=np_X_missing_copy[np_r_PM25_y_mask==-(j+1)]
        # X_missing
        np_X_missing[np_r_PM25_y_mask==-(j+1)]=np.nan
    
    return np_X_missing, np_y_missing, np_r_PM25_y_mask

# a function that returns X_missing, y_missing that are randomly spread
# (instead of chunks. just for comparison)
def return_randomly_spread_X_y(PM25,IL):
 
    X_missing = PM25.copy()
    y_missing = PM25.copy() 
    
    np_PM25 = PM25.values
    
    not_nan_idx = np.argwhere(PM25.notnull().values)    
    test_index = np.random.choice(not_nan_idx.shape[0],IL)
        
    # y_missing 
    y_missing.iloc[:] = np.nan
    np_y_missing = y_missing.values

    # asssign values according to test indexes
    rows, cols = zip(*not_nan_idx[test_index])
    vals = np_PM25[rows, cols]
    np_y_missing[rows, cols] = vals

    # X_missing
    # assign nans according to test indexes
    np_X_missing = X_missing.values
    np_X_missing[rows, cols] = np.nan

    return np_X_missing, np_y_missing

# a function that imputes for each of the methods

def impute_ii_BR(np_X_missing, num_iter):
    imp = IterativeImputer(max_iter=num_iter,estimator=BayesianRidge(),verbose=True) 
    imp.fit(np_X_missing)
    imputed = imp.transform(np_X_missing) 
    return imputed

def impute_ii_RF(np_X_missing, num_iter, rnd_state_forRF, params):
#     n_jobs=-1 # add to ExtraTreesRegressor if I want all cores. 
#     if params['max_depth'] is None:
#         md = None
#     else:
#         md = int(params['max_depth'])
    imp = IterativeImputer(max_iter=num_iter,
                           estimator=ExtraTreesRegressor(n_estimators=int(params['n_estimators']),
#                                                          bootstrap=params['bootstrap'],
                                                         max_depth=params['max_depth'], 
                                                         max_features=params['max_features'],
                                                         min_samples_leaf=int(params['min_samples_leaf']),
                                                         min_samples_split=int(params['min_samples_split']),
                                                         random_state=rnd_state_forRF,
                                                         n_jobs=-1
                                                        ),
                           verbose=False) 
    imp.fit(np_X_missing)
    imputed = imp.transform(np_X_missing)  
    return imputed

def impute_ii_KNN(np_X_missing, num_iter, num_neighbors, weight_type): #'uniform','distance'
#     n_jobs=-1 # add to KNeighborsRegressor if I want all cores. 
    imp = IterativeImputer(max_iter=num_iter,estimator=KNeighborsRegressor(n_neighbors=num_neighbors,weights=weight_type,n_jobs=-1),verbose=True) 
    imp.fit(np_X_missing)
    imputed = imp.transform(np_X_missing)     
    return imputed

def impute_my_KNN(np_X_missing, k):

    imputed = np_X_missing.copy() 
    all_data_mask_nans = imputed.copy()
    mask_nans = np.isnan(imputed)
    all_data_mask_nans[mask_nans]=0

    batch_size = 300
    not_nan_mask = 1-np.isnan(np_X_missing).astype(int)
    # all_data_norm = all_data_mask_nans / np.linalg.norm(all_data_mask_nans,axis=1)[:,np.newaxis] # normalize BY norm
    all_data_norm = np_X_missing / np.nanstd(np_X_missing, axis=0) # normalize BY norm 
#     all_data_norm = (np_X_missing-np.nanmean(np_X_missing, axis=0)) / np.nanstd(np_X_missing, axis=0) # normalize BY norm 

    np_corrMatrix = pd.DataFrame(imputed).corr().values
    
    for cind in tqdm(range(np_X_missing.shape[1])): #tqdm(range(np_X_missing.shape[1]-33)) #range(34,int(np_X_missing.shape[1]/3)+1) #+1=AFULA, +34=all 
        nan_column_mask = np.isnan(np_X_missing[:,cind])
        not_nan_column_mask = np.logical_not(nan_column_mask)

        not_nan_in_colom_all_data_norm = all_data_norm[not_nan_column_mask,:]*np_corrMatrix[:,cind]

        not_nan_not_nan_mask = not_nan_mask[not_nan_column_mask, :] #relevant rows for this column (not nan)

        not_nan_column = np_X_missing[not_nan_column_mask,cind]    
        nan_ind = np.argwhere(nan_column_mask)
        
        for i in tqdm(range(0, len(nan_ind), batch_size), leave=False):
            rinds = np.ravel(nan_ind[i:i+batch_size])
            batch = all_data_norm[rinds, :]*np_corrMatrix[:,cind]

            batch_non_nan_mask = not_nan_mask[rinds, :]
            counts = np.dot(not_nan_not_nan_mask, batch_non_nan_mask.T)

            dists = nan_euclidean_distances(not_nan_in_colom_all_data_norm, batch)

            weights = counts/dists
            min_thr = np.partition(weights,-(k+1),axis=0)[-(k+1),:]
            weights = weights-min_thr
            # !If all the coordinates are missing or if there are no common present coordinates then NaN is returned in dists for that pair!
            weights[(weights<0) | (np.isnan(weights))] = 0
            weights = weights / weights.sum(axis=0)

            values = np.dot(weights.T, not_nan_column)
            imputed[rinds, cind] = values
            
    return imputed
    

# results = []

# for j in range(len(IL)):
#     y_train = np_y_missing[(~np.isnan(np_y_missing)) & (np_r_PM25_y_mask==-(j+1))]
#     y_pred = imputed[(~np.isnan(np_y_missing)) & (np_r_PM25_y_mask==-(j+1))]           

#     # assign results
#     RMSE = np.sqrt(mean_squared_error(y_train, y_pred))
#     MedianAE = median_absolute_error(y_train, y_pred)
#     MeanAE = mean_absolute_error(y_train,y_pred)
#     R2 = r2_score(y_train,y_pred)
#     MeanBias = np.mean(y_pred - y_train) #if positive, we overestimate, if negative we underestimate
#     MedianBias = np.median(y_pred - y_train) #if positive, we overestimate, if negative we underestimate
#     NMB = np.sum(y_pred - y_train)/(np.sum(y_train))# Normalized mean bias
#     NME = np.sum(np.abs(y_pred - y_train))/(np.sum(y_train))# Normalized mean error

#     results.append([IL[j],RMSE,MedianAE,MeanAE,R2,MeanBias,MedianBias,NMB,NME])


# a function that cumputes validation results

def validate(imputed, np_y_missing, np_r_PM25_y_mask, IL):
                           
    all_y_train = np_y_missing[~np.isnan(np_y_missing)]
    all_y_pred = imputed[~np.isnan(np_y_missing)]
    
#     if np.isnan(y_pred).sum():
#         y_train = y_train[~np.isnan(y_pred)]
#         y_pred = y_pred[~np.isnan(y_pred)]
#         print('nan values detected in imputed matrix')
    results = []
      
    # assign results
    RMSE = np.sqrt(mean_squared_error(all_y_train, all_y_pred))
    MedianAE = median_absolute_error(all_y_train, all_y_pred)
    MeanAE = mean_absolute_error(all_y_train,all_y_pred)
    R2 = r2_score(all_y_train,all_y_pred)
    MeanBias = np.mean(all_y_pred - all_y_train) #if positive, we overestimate, if negative we underestimate
    MedianBias = np.median(all_y_pred - all_y_train) #if positive, we overestimate, if negative we underestimate
    NMB = np.sum(all_y_pred - all_y_train)/(np.sum(all_y_train))# Normalized mean bias
    NME = np.sum(np.abs(all_y_pred - all_y_train))/(np.sum(all_y_train))# Normalized mean error

    results.append(['ALL',RMSE,MedianAE,MeanAE,R2,MeanBias,MedianBias,NMB,NME])
         
    for j in range(len(IL)):
        y_train = np_y_missing[(~np.isnan(np_y_missing)) & (np_r_PM25_y_mask==-(j+1))]
        y_pred = imputed[(~np.isnan(np_y_missing)) & (np_r_PM25_y_mask==-(j+1))]           
             
        # assign results
        RMSE = np.sqrt(mean_squared_error(y_train, y_pred))
        MedianAE = median_absolute_error(y_train, y_pred)
        MeanAE = mean_absolute_error(y_train,y_pred)
        R2 = r2_score(y_train,y_pred)
        MeanBias = np.mean(y_pred - y_train) #if positive, we overestimate, if negative we underestimate
        MedianBias = np.median(y_pred - y_train) #if positive, we overestimate, if negative we underestimate
        NMB = np.sum(y_pred - y_train)/(np.sum(y_train))# Normalized mean bias
        NME = np.sum(np.abs(y_pred - y_train))/(np.sum(y_train))# Normalized mean error

        results.append([IL[j],RMSE,MedianAE,MeanAE,R2,MeanBias,MedianBias,NMB,NME])
       
    results = pd.DataFrame(results, columns=['IL','RMSE','MedianAE','MeanAE','R2','MeanBias','MedianBias','NMB','NME'])   
    return results



def simple_validation(imputed, np_y_missing):
    y_train = np_y_missing[~np.isnan(np_y_missing)]
    y_pred = imputed[~np.isnan(np_y_missing)]
    
    if np.isnan(y_pred).sum():
        y_train = y_train[~np.isnan(y_pred)]
        y_pred = y_pred[~np.isnan(y_pred)]
        print('nan values detected in imputed matrix')
      
    # assign results
    RMSE = np.sqrt(mean_squared_error(y_train, y_pred))
#     MedianAE = median_absolute_error(y_train, y_pred)
#     MeanAE = mean_absolute_error(y_train,y_pred)
#     R2 = r2_score(y_train,y_pred)
#     MeanBias = np.mean(y_pred - y_train) #if positive, we overestimate, if negative we underestimate
#     MedianBias = np.median(y_pred - y_train) #if positive, we overestimate, if negative we underestimate
#     NMB = np.sum(y_pred - y_train)/(np.sum(y_train))# Normalized mean bias
#     NME = np.sum(np.abs(y_pred - y_train))/(np.sum(y_train))# Normalized mean error

#     results = [RMSE,MedianAE,MeanAE,R2,MeanBias,MedianBias,NMB,NME]
    
    return RMSE


def plot_hist(PM25, imputed, np_X_missing, density,no_bins,option):
#     # option-1
    if option=='1':
        plt.hist([PM25[~np.isnan(PM25)], np_X_missing[~np.isnan(np_X_missing)], imputed[np.isnan(np_X_missing)]],bins=1000,label=['original PM25','original X','imputed'],density=True)
    if option=='2':
        plt.hist(PM25[~np.isnan(PM25)],bins=no_bins,label=['original PM25'],alpha=0.3,density=density) #,edgecolor="k"
        plt.hist(np_X_missing[~np.isnan(np_X_missing)],bins=no_bins,label=['original X'],alpha=0.3,density=density)
        plt.hist(imputed[np.isnan(np_X_missing)],bins=no_bins,label=['imputed'],alpha=0.3,density=density)
    plt.xlim([-20,100])
    plt.legend()
    plt.show();

# a function for box plots for results


# statistics functions
def get_nan_lengths(PM25,nanLengths):
    
    if nanLengths:
        PM25[PM25.notnull()] = 1
        PM25[PM25.isnull()] = 0
    else:
        PM25[PM25.notnull()] = 0
        PM25[PM25.isnull()] = 1
        
    diffs = PM25.diff(axis = 0)

    missing_interval_lengths = []

    for monitor in range(PM25.shape[1]):

        #begining
        if PM25.iloc[0,monitor]==0:
            diffs.iloc[0,monitor]=-1

        #end
        if PM25.iloc[-1,monitor]==0:
            diffs.iloc[-1,monitor]=1

        row_start = np.where(diffs.iloc[:,monitor] == -1)[0]
        row_end = np.where(diffs.iloc[:,monitor] == 1)[0]

        xranges = list(list(zip(row_start,row_end-row_start)))
        missing_interval_lengths.extend(row_end-row_start)

    return missing_interval_lengths


def plot_nan_lengths_dist(missing_interval_lengths):
     
    data = [i*30/60 for i in missing_interval_lengths]#in hours
    _, bins = np.histogram(np.log10(data), bins='auto')

    # bins are unequal in width in a way that would make them look equal on a logarithmic scale.
    # matplotlib histogram
    plt.hist(data, color = 'blue',bins=10**bins)
    plt.gca().set_xscale("log")

    plt.title('distribution of length of \nmissing data intervals, by count')
    plt.xlabel('Interval length [hours]')
    plt.ylabel('Percentage [%]\n')
    plt.yticks(ticks=np.arange(0,len(missing_interval_lengths),np.floor(len(missing_interval_lengths)/10)),labels=np.round(100*np.arange(0,len(missing_interval_lengths),np.floor(len(missing_interval_lengths)/10))/len(data),1))
    
    # plt.ylim(0,1000)
    # plt.xlim(0,100)
    plt.tight_layout()
    plt.rcParams.update({'font.size': 14})

    plt.show();
    
def plot_nan_lengths_cum_dist(missing_interval_lengths):    
    # pd.Series(missing_interval_lengths).hist(weights=missing_interval_lengths, bins=100, cumulative=True)
    a,b = np.histogram(missing_interval_lengths,weights=missing_interval_lengths, bins=np.max(missing_interval_lengths))
    plt.plot(a.cumsum()/np.sum(a));
    plt.plot(np.ones(np.max(missing_interval_lengths))*0.5,'r');


### initialize

# originally I've set this threshold to 0.6, but then I realized that I should take all stations first and 
# only choose relevant stations at the end, according to the imputation results (and maybe original missingness rate)

threshold = 0.6 # how much non-missing values are in the time-series in order to include the station?
start_year = '2012'
times, r_PM25 = initialize(threshold, start_year)

times.shape
r_PM25.shape

r_PM25

# times_df = pd.DataFrame(times)
# times_df.to_csv("~/Documents/saved_data_from_notebooks/times_df.csv")

# (0) Add two years of missing data in ASHKELON_SOUTH (2012-2013)

# import json
# from pandas.io.json import json_normalize
# import requests

# # extraction is in UTC time. it gives you results for UTC time. 
# # the true corresponding time in Israel it was 2 or 3 hours later
# def get_data(from_date, to_date, stationId,myToken):
#     myUrl = 'https://api.svivaaqm.net/v1/envista/stations/'+stationId+'/data?from='+from_date+'&to='+to_date
#     head = {'Authorization': 'ApiToken {}'.format(myToken), 'envi-data-source': 'MANA'}
#     response = requests.get(myUrl, headers=head)
#     return response

# # use this to get column names and units of the station
# def get_column_names(stationId):
#     myUrl = 'https://api.svivaaqm.net/v1/envista/stations/'+stationId+'?from=2019-12-01T00:00&to=2019/12/01T00:06'
#     head = {'Authorization': 'ApiToken {}'.format(myToken), 'envi-data-source': 'MANA'}
#     response = requests.get(myUrl, headers=head)
#     # units
#     extract = response.json()['monitors']
#     units = dict()
#     for e in extract:
#         units.update({e['name']:e['units']})
#     column_names = [i + ' [' + j +']' for i, j in units.items()]
#     return list(units.keys()), column_names


# def get_dataFrame(dict_train,stationId):
#     dates = []
#     for row in dict_train:
#         dates.append(row['datetime'])

#     pollutants = []
#     for row in dict_train:
#         pollutant = dict()
#         for p in row['channels']:
#             pollutant.update({p['name']:p['value']})
#         pollutants.append(pollutant)

#     cols, station_columns = get_column_names(str(stationId))

#     total_list = []
#     for c in cols: #number of columns(j)
#         vals = []
#         for p in pollutants: #number of rows or values(i)
#             if p[c] is not None:
#                 vals.append(p[c])
#             else:
#                 vals.append(np.NaN)
#         total_list.append(vals)

#     data_df = pd.DataFrame(np.transpose(total_list), index = pd.to_datetime(dates,utc=True).tz_convert('Israel'), columns = station_columns)
    
#     return data_df

# # example
# myToken = '71e67c41-8478-4310-9293-196f559493ca'
# myUrl = 'https://api.svivaaqm.net/v1/envista/stations?from=2012-01-01T00:00&to=2012-01-01T01:00'
# # myUrl = 'https://www.svivaaqm.net:44301/v1/envista/stations/452?from=2019-11-01T00:00&to=2019/11/01T00:12'
# head = {'Authorization': 'ApiToken {}'.format(myToken), 'envi-data-source': 'MANA'}
# response = requests.get(myUrl, headers=head)

# j_response = response.json()
# j_response

# idx = r_PM25['ASHKELON_SOUTH'].first_valid_index()
# # r_PM25.loc[:35087,'ASHKELON_SOUTH']
# # times[:35087]
# # origin_ashk = r_PM25['ASHKELON_SOUTH'].values

# r_PM25.loc[:idx,'ASHKELON_SOUTH']
# times[:idx]

# myToken = '71e67c41-8478-4310-9293-196f559493ca'

# from_time = '2012-01-01T00:00'
# to_time = '2014-01-01T00:05'

# stationId = 160
# new_ashk = get_data(from_time, to_time,str(stationId),myToken)

# new_ashk

# dict_train = new_ashk.json()['data']
# df_temp = get_dataFrame(dict_train,stationId)

# df_temp['PM2.5 [µg/m³]'].to_pickle('pm25_ashkelon_south_2012_2013.pkl')

temp1 = pd.read_pickle('pm25_ashkelon_south_2012_2013.pkl')

temp1.resample('30T',closed= 'right').mean().round(1)

idx = r_PM25['ASHKELON_SOUTH'].first_valid_index()
r_PM25.loc[:idx,'ASHKELON_SOUTH']=temp1.resample('30T',closed= 'right').mean().round(1).values

r_PM25.loc[:idx,'ASHKELON_SOUTH']
times[:idx]



# (0) compare with other methods

# https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779

# fast KNN (with KDE)

# sys.setrecursionlimit(100000) #Increase the recursion limit of the OS

# start the KNN training
# imputed_training=fast_knn(r_PM25.values, k=10)

# MICE

# imputed_training=mice(r_PM25.values)

# (0) find charecteristic lag times

# # Autocorrelation
# from pandas import datetime
# from pandas.plotting import autocorrelation_plot
# from statsmodels.graphics.tsaplots import plot_pacf
 
# series = r_PM25.loc[200:220,'ANTOKOLSKY'].values
# # autocorrelation_plot(series)
# plot_pacf(series)
# # plt.ylim([-0.05,0.05])

# # I think that lag1 and lag2 is enough. But seems like I have to find a way to conclude it on all, but it is very unnecessary



# (1) set aside a hold-out set (test set)

# what chunk sizes? (in hours)
IL = [720,120,24,6,1,0.5] # 1 month, 5 days, 1 day, 6 hours, 1 hour, half hour

np.random.seed(2) # if want to reproduce. 

# drop the following stations as it is impossible to set a test set aside with them
# with zero threshold:
# r_PM25.drop(['AKO','KVISH9','BNEI_DAROM','YAD_BINYAMIN','TEL_HAY'],axis=1,inplace=True)
# with 0.2 threshold:
# r_PM25.drop(['AKO','BNEI_DAROM'],axis=1,inplace=True)

# r_PM25.columns[19]
# r_PM25.drop(['AKO'],axis=1,inplace=True)

# r_PM25.columns[35]
# r_PM25.drop(['KVISH9'],axis=1,inplace=True)

# r_PM25.columns[39]
# r_PM25.drop(['BNEI_DAROM'],axis=1,inplace=True)

# r_PM25.columns[53]
# r_PM25.drop(['YAD_BINYAMIN'],axis=1,inplace=True)

# r_PM25.columns[58]
# r_PM25.drop(['TEL_HAY'],axis=1,inplace=True)

np_X_missing, np_y_missing, np_r_PM25_y_mask = return_X_y(r_PM25,IL)
# # for comparison - small chunks randomly spread
# np_X_missing, np_y_missing = return_randomly_spread_X_y(r_PM25,1440*34)

np.min(100*1440*6/r_PM25.notnull().sum())
np.max(100*1440*6/r_PM25.notnull().sum())
np.mean(100*1440*6/r_PM25.notnull().sum())

# average of holdout set percentage from non-missing data
100*1440*6*r_PM25.shape[1]/r_PM25.notnull().sum().sum()

# Only label every 20th value
ticks_to_use = np.arange(0,r_PM25.shape[0]-365*24,366*48)
# Set format of labels (note year not excluded as requested)
labels = [times[i].strftime("%Y") for i in ticks_to_use]

# plot the missing value chunks! see it is well distributed
X_missing = pd.DataFrame(np_X_missing,columns=r_PM25.columns)
y_missing = pd.DataFrame(np_y_missing,columns=r_PM25.columns)

# ax,fig = plt.subplots(figsize=(20,20))
# sns.heatmap(y_missing.iloc[:,:75].notnull(), cbar=False);
# # sns.heatmap(X_missing.iloc[:,:75].isnull(), cbar=False);

# # Now set the ticks and labels
# plt.yticks(ticks_to_use,labels)

# plt.rcParams.update({'font.size': 18})
# plt.show();



# (2) hyperparameter tunning by 3-fold CV 

#CONCLUSION - NO NEED FOR "REGULAR CV", JUST DO THE HYPERPARAMETER TUNNING WITH MY CV AND TEST ON A TEST SET I SHOULD 
#PUT ASIDE

# https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a

# first find optimal parameters with only 1 iteration, then optimize for number of iterations. 
# Possibly - the number of iterations are related to the data of validation... :/ but it's too much to account for that too

# hyperParams
# RF - number of trees (probably the more the better),max_depth...
# myKNN - number of neighbors, distance=Euclidean

# in IterativeImputer - the method of initial imputation - mean/median...

# CV is needed here to ensure that the decision is done after seeing the whole data. 
# "Under cross validation, each point gets tested exactly once, which seems fair. 
# However, cross-validation only explores a few of the possible ways that your data 
# could have been partitioned. Monte Carlo cross validation can give you a less variable, 
# but more biased estimate"
# https://stats.stackexchange.com/questions/51416/k-fold-vs-monte-carlo-cross-validation

# assumption - the model that would perform best in a 'regular CV' (interval lengths are short), would perform
# best on long intervals as well. 

#### RandomSearchCV not suitable, use instead Hyperopt

#### Hyperopt for ExtraTree

# r_PM25, X_missing, y_missing
# it's important to tune the parameters on X_missing = data after extraction of a test set (=y_missing)
# and not on r_PM25!

'''As a brief primer, Bayesian optimization finds the value that minimizes an objective function
by building a surrogate function (probability model) based on past evaluation results of the objective.
The surrogate is cheaper to optimize than the objective, so the next input values to evaluate are selected
by applying a criterion to the surrogate (often Expected Improvement). Bayesian methods differ from random
or grid search in that they use past evaluation results to choose the next values to evaluate. 
The concept is: limit expensive evaluations of the objective function by choosing the next input values
based on those that have done well in the past.'''
'''The aim is to find the hyperparameters that yield the lowest error on the validation set in the hope
that these results generalize to the testing set.'''

# Bergstra et al - http://proceedings.mlr.press/v28/bergstra13.pdf

# https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a
# http://hyperopt.github.io/hyperopt/
# https://github.com/hyperopt/hyperopt/wiki/FMin

from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import pyll
from hyperopt import Trials
from hyperopt import fmin


# I want to control the randomness of CV because its the randomness of the split. that way, the split is always the same. 
# I also want to control the randomness of RF (which is the shuffle of the order of features) so the comparison in 
# the other parameters can be done. 
rnd_state_forCV = 1
rnd_state_forRF = 0 # ensure that your performance is not affected by the random initial state. https://stackoverflow.com/questions/55070918/does-setting-a-random-state-in-sklearns-randomforestclassifier-bias-your-model

N_FOLDS = 5
num_iter = 1 #just for the optimization

PM25_opt = X_missing.copy()

not_nan_idx = np.argwhere(X_missing.notnull().values) #X_missing is after removing a final test set
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=rnd_state_forCV)

# the objective function calculates the average performance (based on k-fold cv)

def objective(params):
    
    
    # Perform n_fold cross validation with hyperparameters
    # Evaluate based on RMSE
    # Still not sure about the randomness of CV. It seems that gridSearch and randomSearch keeps the same devision of the kfold and evaluate all models on them. 
  
    total_loss = []
    print(params)    
    
    for k, (train_index, test_index) in enumerate(kf.split(not_nan_idx)):
        
        X_missing = PM25_opt.copy()
        y_missing = PM25_opt.copy()
        np_PM25_opt = PM25_opt.values
        
        # y_missing 
        y_missing.iloc[:] = np.nan
        np_y_missing = y_missing.values

        # asssign values according to test indexes
        rows, cols = zip(*not_nan_idx[test_index])
        vals = np_PM25_opt[rows, cols]
        np_y_missing[rows, cols] = vals

        # X_missing
        # assign nans according to test indexes
        np_X_missing = X_missing.values
        np_X_missing[rows, cols] = np.nan
       
        # impute
        imputed = impute_ii_RF(np_X_missing, num_iter, rnd_state_forRF, {**params, 'n_estimators':100})
        # validate
        #!!!take only RMSE!!! (edit simple_validation)
        loss = simple_validation(imputed, np_y_missing)
        total_loss.append(loss)
        
    # Write to the csv file ('a' means append)
    with open(out_file, 'a') as of_connection:
        writer = csv.writer(of_connection)
        writer.writerow([total_loss, params])


    # Dictionary with information for evaluation
    return {'loss': np.mean(total_loss), 'params': params, 'status': STATUS_OK}

# Define the search space

# ne = [10,30,50,70,100,120,150,200,300,400,500] #default 100
# mf = ['auto', 'sqrt', 'log2'] #default 'auto'
# md = [3, 4, 5, 6, 7, 8, 9, 10, 50, 100, None] #default None (nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples)
# mss = [2, 3, 4, 5, 10, 20] #default 2
# msl = [1, 2, 3, 4, 5, 10, 20] #default 1


space = {
    #'n_estimators': hp.quniform('n_estimators',30,420,30),# Discrete uniform distribution
    'max_features': hp.choice('max_features',['auto', 'sqrt', 'log2']),
    'max_depth': hp.choice('max_depth',[None, hp.qlognormal('ExtraTree_max_depth', 3, 1, 1)]),
#    'min_samples_split': hp.qlognormal('min_samples_split', 2, 1, 1),
#    'min_samples_leaf': hp.qlognormal('min_samples_leaf', 1, 1, 1)
    'min_samples_split': hp.quniform('min_samples_split',2,20,1),
    'min_samples_leaf': hp.quniform('min_samples_leaf',1,20,1)
#     'bootstrap': hp.choice('bootstrap', [True, False])
}

# Sample from the full space
params = pyll.stochastic.sample(space)
print(params)

# params['bootstrap'] #I think bootstrap makes sense only if max_samples is not all samples... 
#params['n_estimators']
params['max_features']
params['max_depth']
params['min_samples_split']
params['min_samples_leaf']

tpe_algorithm = tpe.suggest

# Trials object to track progress
bayes_trials = Trials()

params

# writing to a CSV
import csv

# File to save first results
out_file = 'hyperOpt_trials_iiRF.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)

# Write the headers to the file
writer.writerow(['loss', 'params'])
of_connection.close()

MAX_EVALS = 500

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest,
            max_evals = MAX_EVALS, trials = bayes_trials)
