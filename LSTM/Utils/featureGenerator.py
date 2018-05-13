'''
Generates features from Order Book data
Data format:
    - data_dir folder contains CSVs with order book data for a given stock
    (msft, goog, amzn, aapl) at a given depth (1-10)
    - Order book data is second-by-second
'''
import pandas as pd
import numpy as np
import os
import sys
'''
Raw Data Format:
    columns (n=1:10):
        - datetime (YYYY-MM-DD H:M:S)
        - bid{n}
        - ask{n}
        - bsize{n}
        - asize{n}
        - bnum{n}
        - anum{n}
        - vwap
        - notional
        - volume
        - last_price
        - mid
        - spread
        - wmid
        - last_size
        - last_SRO
'''

data_dir = '../../ProjectData/'


def mergeOrderBookDays(data_dir, out_path, prefixes):
    '''
    Raw data is separated into files by date (e.g. msft-20170417)
    Merge files with the same prefix (ticker) by day
    '''
    files = os.listdir(data_dir)

    for prefix in prefixes:
        data = pd.DataFrame()
        fnames = [f for f in files if (str(f).startswith(prefix) and str(f).endswith('orderbook.csv'))]
        print(fnames)
        for f in fnames:
            print(data_dir + str(f))
            data = data.append(pd.read_csv(data_dir + str(f)))
        data.to_csv(out_path, index = False)
    #return data

def createResponseVariable(data, response_type = 'Classification', look_forward = 0):
    '''
    Generates response variable for raw input data.
    Response variable will be:
        - mid price of next tick order book (response_type = 'Regression')
        - Up , Down, No Change (response_type = 'Classification')
    '''
    if response_type.upper() == 'REGRESSION':
        response_col = [data.loc[i+1, 'direct.vwap'] for i in range(len(data)-1)]

    elif response_type.upper() == 'CLASSIFICATION':
        #response_col = [data.loc[i+1, 'direct.vwap'] for i in range(len(data)-(1))]
        
        response_col = []
        for i in range(len(data)-1):
            current_price = data.loc[i, 'direct.vwap']
            next_price = data.loc[i+1, 'direct.vwap']
            diff = next_price - current_price
            if diff == 0:
                response_col.append(0)
            elif diff > 0:
                response_col.append(1)
            elif diff < 0:
                response_col.append(2)
        
    data = data[:-1] # get rid of last row, which won't have a response variable
    data['Response'] = response_col
    return data

def calculateImbalance(data):
    '''
    Calulate Order Book imbalance
    '''
    pass



def calculatePriceDiffs(data):
    '''
    (V2)

    1. P_ask_n - P_ask_1 (ask diff)
    2. P_bid_n - P_bid_1 (bid diff)
    3. abs(P_ask_i+1 - P_ask_i) {i = 1 : n}
    4.
    '''
    # 1. Ask Diff



def calculateMeanPricesAndVolumes(data):
    '''
    (V4)
    Mean Bid/Ask, Prices/Volumes
    sum(Price_i)/n
    sum(Volumes_i)/n
    '''
    #data['meanBid'] = 'NA'
    bid_col_list = ['direct.bid{}'.format(i) for i in range(1,11)]
    data['meanBid'] = data[bid_col_list].sum(axis=1)/10

    #data['meanAsk'] = 'NA'
    ask_col_list = ['direct.ask{}'.format(i) for i in range(1,11)]
    data['meanAsk'] = data[ask_col_list].sum(axis=1)/10

    #data['meanBidNum'] = 'NA'
    bidNum_col_list = ['direct.bnum{}'.format(i) for i in range(1,11)]
    data['meanBidNum'] = data[bidNum_col_list].sum(axis=1)/10

    #data['meanAskNum'] = 'NA'
    askNum_col_list = ['direct.anum{}'.format(i) for i in range(1,11)]
    data['meanAskNum'] = data[askNum_col_list].sum(axis=1)/10


    bidVol_col_list = ['direct.bsize{}'.format(i) for i in range(1,11)]
    data['meanBidVol'] = data[bidVol_col_list].sum(axis=1)/10

    askVol_col_list = ['direct.asize{}'.format(i) for i in range(1,11)]
    data['meanAskVol'] = data[askVol_col_list].sum(axis=1)/10

    #var_cols = bid_col_list + ask_col_list + bidNum_col_list + askNum_col_list +bidVol_col_list+askVol_col_list
    var_cols = ['meanBid', 'meanAsk', 'meanBidNum', 'meanAskNum', 'meanBidVol','meanAskVol']
    return data, var_cols

def calculateSpreadsAndMidPrices(data):
    '''
    (V2)
    bid-ask spreads and mid-prices

    P_ask - P_bid{i=1,...,n}
    P_ask + P_bid{i=1,...,n}
    '''
    # calculate spreads
    for i in range(1,11):
        spread = [data.loc[j, 'direct.ask{}'.format(i)] - data.loc[j, 'direct.bid{}'.format(i)] for j in range(len(data))]
        data['spread_{}'.format(i)] = spread

    # calculate mid prices
    for i in range(1,11):
        midPrice = [(data.loc[j, 'direct.ask{}'.format(i)] + data.loc[j, 'direct.bid{}'.format(i)])/2 for j in range(len(data))]
        data['midPrice_{}'.format(i)] = midPrice

    var_cols_spread = ['spread_{}'.format(i) for i in range(1,11)]
    var_cols_mp = ['midPrice_{}'.format(i) for i in range(1,11)]
    var_cols = var_cols_spread + var_cols_mp
    return data, var_cols

def calculateAccumulatedDifferences(data):
    '''
    (V5 (= V7 ?))
    sum(P_ask_i - P_bid_i)
    sum(V_ask_i - V_bid_i)
    '''
    askPrice_cols = ['direct.ask{}'.format(i) for i in range(1,11)]
    bidPrice_cols = ['direct.bid{}'.format(i) for i in range(1,11)]
    data['accumulatedPriceDiff'] = data[askPrice_cols].sum(axis=1) - data[bidPrice_cols].sum(axis=1)

    askVolume_cols = ['direct.asize{}'.format(i) for i in range(1,11)]
    bidVolume_cols = ['direct.bsize{}'.format(i) for i in range(1,11)]
    data['accumulatedVolumeDiff'] = data[askVolume_cols].sum(axis=1) - data[bidVolume_cols].sum(axis=1)

    var_cols = ['accumulatedPriceDiff', 'accumulatedVolumeDiff']
    return data, var_cols

def createFeatures(data_path, out_path, response_type, look_forward):
    '''
    Generates features from Order Book Data

    Inputs:
        - data_path: path to order book data
        - out_path: path to place generated file
    Output:
        - featureMatrix: data frame containing original data and features
    '''
    askPrice_vars = ['direct.ask{}'.format(i) for i in range(1,11)]
    bidPrice_vars = ['direct.bid{}'.format(i) for i in range(1,11)]
    askVolume_vars = ['direct.asize{}'.format(i) for i in range(1,11)]
    bidVolume_vars = ['direct.bsize{}'.format(i) for i in range(1,11)]
    orig_vars = askPrice_vars + bidPrice_vars +  askVolume_vars + bidVolume_vars

    data = pd.read_csv(data_path)
    data, meanPriceVol_vars = calculateMeanPricesAndVolumes(data) # V4
    data, spreadMidPrice_vars = calculateSpreadsAndMidPrices(data) # V2
    data, accumlatedDiff_vars = calculateAccumulatedDifferences(data) # V5
    data = createResponseVariable(data, response_type, look_forward)

    # vwap = V6
    feature_vars =  ['direct.vwap'] + orig_vars + meanPriceVol_vars + spreadMidPrice_vars + accumlatedDiff_vars + ['Response']
    feature_vars = ['direct.vwap'] + meanPriceVol_vars + accumlatedDiff_vars + spreadMidPrice_vars + ['Response']
    data = data[feature_vars]
    data.to_csv(out_path, index = False)
