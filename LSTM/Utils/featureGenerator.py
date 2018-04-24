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

def createResponseVariable(data, response_type = 'Classification'):
    '''
    Generates response variable for raw input data.
    Response variable will be:
        - mid price of next tick order book (response_type = 'Regression')
        - Up , Down, No Change (response_type = 'Classification')
    '''
    if response_type.upper() == 'REGRESSION':
        response_col = [data.loc[i+1, 'direct.mid'] for i in range(len(data)-1)]

    elif response_type.upper() == 'CLASSIFICATION':
        response_col = []
        for i in range(len(data)-1):
            current_price = data.loc[i, 'direct.mid']
            next_price = data.loc[i+1, 'direct.mid']
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

def calculateMeanPricesAndVolumes(data):
    '''
    Mean Bid/Ask, Prices/Volumes
    sum(Price_i)/n
    '''
    data['meanBid'] = 'NA'
    bid_col_list = ['direct.bid{}'.format(i) for i in range(1,11)]
    print(bid_col_list)
    #for i in range(len(data)):
    data['meanBid'] = data[bid_col_list].sum(axis=1)

    data['meanAsk'] = 'NA'
    ask_col_list = ['direct.ask{}'.format(i) for i in range(1,11)]
    #for i in range(len(data)):
    data['meanAsk'] = data[ask_col_list].sum(axis=1)

    data['meanBidNum'] = 'NA'
    bidNum_col_list = ['direct.bnum{}'.format(i) for i in range(1,11)]
    #for i in range(len(data)):
    data['meanBidNum'] = data[bidNum_col_list].sum(axis=1)

    data['meanAskNum'] = 'NA'
    askNum_col_list = ['direct.anum{}'.format(i) for i in range(1,11)]
    #for i in range(len(data)):
    data['meanAskNum'] = data[askNum_col_list].sum(axis=1)

    var_cols = bid_col_list + ask_col_list + bidNum_col_list + askNum_col_list
    var_cols += ['meanBid', 'meanAsk', 'meanBidNum', 'meanAskNum']
    return data, var_cols

def calculateSpreadsAndMidPrices(data):
    '''
    bid-ask spreads and mid-prices

    P_ask - P_bid{i=1,...,n}
    P_ask + P_bid{i=1,...,n}
    '''
    # calculate spreads
    for i in range(1,11):
        #data['spread_{}'.format(i)] = 'NA'
        #bid = data.loc[i, 'direct.bid{}'.format(i)]
        #ask = data.loc[j, 'direct.ask{}'.format(i)]
        spread = [data.loc[j, 'direct.ask{}'.format(i)] - data.loc[j, 'direct.bid{}'.format(i)] for j in range(len(data))]
        data['spread_{}'.format(i)] = spread

    # calculate mid prices
    for i in range(1,11):
        #data['midPrice_{}'.format(i)] = 'NA'
        #bid = data.loc[j, 'direct.bid{}'.format(i)]
        #ask = data.loc[j, 'direct.ask{}'.format(i)]
        midPrice = [(data.loc[j, 'direct.ask{}'.format(i)] + data.loc[j, 'direct.bid{}'.format(i)])/2 for j in range(len(data))]
        data['midPrice_{}'.format(i)] = midPrice

    var_cols_spread = ['spread_{}'.format(i) for i in range(1,11)]
    var_cols_mp = ['midPrice_{}'.format(i) for i in range(1,11)]
    var_cols = var_cols_spread + var_cols_mp
    return data, var_cols

def createFeatures(data_path, out_path, response_type):
    '''
    Generates features from Order Book Data

    Inputs:
        - data_path: path to order book data
        - out_path: path to place generated file
    Output:
        - featureMatrix: data frame containing original data and features
    '''

    data = pd.read_csv(data_path)
    data, meanPriceVol_vars = calculateMeanPricesAndVolumes(data)

    data, spreadMidPrice_vars = calculateSpreadsAndMidPrices(data)
    #data = calculateImbalance(data)
    data = createResponseVariable(data, response_type)

    feature_vars = meanPriceVol_vars + spreadMidPrice_vars + ['Response']
    data = data[feature_vars]
    data.to_csv(out_path, index = False)
