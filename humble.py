'''
deployment helper file for humble stock model
'''

import pandas as pd
from pandas import DatetimeIndex
from pandas.plotting import table
import datetime 
from datetime import datetime, timedelta
import numpy as np
from pickle import load, dump
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import unique_labels


def Preprocess(file):
    '''
    import downloaded csv file into pandas
    format dates and columns
    scale to adjusted close
    compute daily returns based on close to close
    return preprocessed df
    '''
    # import and format with sort ascending
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date', ascending=True)
    df = df.reset_index(drop=True)
    df = df.rename(index=str, columns={'Date':'date',
                                       'Open': 'open', 
                                       'High': 'high',
                                       'Low': 'low',
                                       'Close':'close',
                                       'Adj Close':'adj_close',
                                       'Volume':'volume',
                                       })
    # clean
    df = df.dropna()
    df = df.drop_duplicates()
    
    # scale
    df_scale = pd.DataFrame()
    close = df.close.to_numpy().astype(float)
    adj = df.adj_close.to_numpy().astype(float)
    scale = adj / close
    df_scale['date'] = df['date'].copy()
    df_scale['open']=df.open.to_numpy().astype(float)*scale
    df_scale['high']=df.high.to_numpy().astype(float)*scale
    df_scale['low']=df.low.to_numpy().astype(float)*scale
    df_scale['close']=df.close.to_numpy().astype(float)*scale
    df_scale['volume']=df.volume.to_numpy().astype(int)
    
    # log returns
    prices = df_scale['close']
    lookback = 1
    daily_returns = np.log(prices) - np.log(prices.shift(lookback))
    df_scale['log_ret'] = daily_returns

    return df_scale


def ComputeUnitShape(prices, sigmas, dayspan):
    '''
    compute one day shape
    '''
    abs_deltas = (prices) - (prices.shift(dayspan))
    s_ratios = abs_deltas / sigmas
    ups = 3*(s_ratios>1)
    downs = 1*(s_ratios<-1)
    neuts = 2*((s_ratios>=-1)&(s_ratios<=1))      
    return (ups+downs+neuts)


def ComputePentaShape(unitshape, dayspan):
    '''
    import unit shape series and dayspan
    compute 5-period shape ordinals
    return penta shape series
    '''
    ago5s = 10000*(unitshape.shift(4*dayspan))
    ago4s = 1000*(unitshape.shift(3*dayspan))
    ago3s = 100*(unitshape.shift(2*dayspan))
    ago2s = 10*(unitshape.shift(1*dayspan))
    return (ago5s+ago4s+ago3s+ago2s+unitshape)


def BuildFeatures(df):
    '''
    import preprocessed market df
    compute statistics 
    use unit shape spans of 1, 3, 5, 7, 9 days
    build penta shape ordinals from unit shapes
    return augmented df
    '''
    df_for = df.copy()
    
    # raw data overlaps
    shifts = [['o1','h1','l1','c1'],
              ['o2','h2','l2','c2'],
              ['o3','h3','l3','c3'],
              ['o4','h4','l4','c4'],
              ['o5','h5','l5','c5'],
              ['o6','h6','l6','c6'],
              ['o7','h7','l7','c7'],
              ['o8','h8','l8','c8'],
             ]
    # format df to calculate price estimates and standard deviations
    # shift old figures up to current
    for j, shift in zip(range(1,9),shifts):
        df_for[shift[0]] = df_for.open.shift(j)
        df_for[shift[1]] = df_for.high.shift(j)
        df_for[shift[2]] = df_for.low.shift(j)
        df_for[shift[3]] = df_for.close.shift(j)

    # define price estimate columns for 1,3,5,7,9 past day spans
    p1_col = df_for.loc[:,"open":"close"].astype(float)
    p3_col = df_for.loc[:,"open":"c2"].astype(float)
    p5_col = df_for.loc[:,"open":"c4"].astype(float)
    p7_col = df_for.loc[:,"open":"c6"].astype(float)
    p9_col = df_for.loc[:,"open":"c8"].astype(float)
    p_cols = [p1_col, p3_col, p5_col, p7_col, p9_col]

    # compute price estimates and standard deviations for spans
    stats = [['pe1','sd1'],['pe3','sd3'],['pe5','sd5'],
             ['pe7','sd7'],['pe9','sd9']]
    for stat, p_col in zip(stats, p_cols):
        df_for[stat[0]] = p_col.mean(axis=1)
        df_for[stat[1]] = p_col.std(axis=1)

    # leave some raw data behind
    df_prep = df_for[['date','open','high','low','close','volume',
                      'log_ret','pe1','sd1','pe3','sd3','pe5','sd5',
                      'pe7','sd7','pe9','sd9']].copy()
    
    # add lookback unit shapes to df
    unitshapes = ['ds1','ds3','ds5','ds7','ds9']
    dayspans = [1,3,5,7,9]
    for shape, stat, span in zip(unitshapes, stats, dayspans):
        df_prep[shape] = ComputeUnitShape(df_prep[stat[0]], 
                                          df_prep[stat[1]], span)
        
    # add lookback penta shapes to df
    shapes = ['shp1','shp3','shp5','shp7','shp9']
    for shape, unitshape, span in zip(shapes, unitshapes, dayspans):
        df_prep[shape] = ComputePentaShape(df_prep[unitshape], span)

    #trim the head and tail NaN's off then format
    # note that five lookbacks of span nine is 45 day rows on the head
    # note that seven look aheads of span one is 7 day rows on the tail
    df_trim = df_prep[45:].copy()
    df_trim[['shp1','shp3','shp5','shp7','shp9']] = \
    df_trim[['shp1','shp3','shp5','shp7','shp9']].astype(int)
    
    return df_trim


def OneHot(df):
    '''
    input data with ordinal features
    one hot encode ordinals
    return encoded data
    '''
    df_mkt = df.copy()
    # One-Hot encode the non-numerical categorical variables
    cat_vars = ['ds1', 'ds3', 'ds5', 'ds7', 'ds9', 
                'shp1', 'shp3', 'shp5', 'shp7', 'shp9']
    for var in  cat_vars:
        # for each cat add dummy var, drop original column
        df_mkt = pd.concat([df_mkt.drop(var, axis=1), 
                            pd.get_dummies(df_mkt[var], 
                                           prefix=var, 
                                           prefix_sep='_', 
                                           drop_first=True)], axis=1)
    return df_mkt 


def test_train_split(df_mkt, val_year, num_ty):
    '''
    input market data, validation year, and number of train years
    split preprocessed data into train, test, and validation dataframes
    validation year comes after test year
    test year comes after years of train data
    return train, test, and validation dataframes
    '''
    df = df_mkt.copy()
    years = df.date.map(lambda x: x.strftime('%Y')).astype(int)
    
    #train = years < test_year for 3 years behind
    test_year = val_year - 1
    train = ((test_year-num_ty <= years) & (years < test_year))
    test = np.isin(years, test_year)
    val = np.isin(years, val_year)

    df_train = df[train].copy()
    df_test = df[test].copy()
    df_val = df[val].copy()
    
    return df_train, df_test, df_val


def Thresh(X_data, model, thresh):
    '''
    import X_data, model, and threshold value
    return predictions after thresholding
    '''
    m_pred = model.predict_proba(X_data)[:,1]
    t_pred = [1 if (prob >= thresh) else 0 for prob in m_pred]
    return t_pred

def Pipe(file):
    '''
    import file
    pipeline process
    augment predictions
    return df with buy predictions
    '''
    # load the model
    feat = load(open('feature.pkl', 'rb'))
    # load the model
    mod = load(open('model.pkl', 'rb'))
    # load the scaler
    scl = load(open('scaler.pkl', 'rb'))
    # preprocess
    dfpre = Preprocess('IBM.csv')
    # add features
    dfft = BuildFeatures(dfpre)
    # one hot encode
    dfp = OneHot(dfft)
    # grab the last year
    _, _, dfval = test_train_split(dfp, 2020, 5)
    # scale features
    dfval[feat] = scl.transform(dfval[feat])
    # threshold tuned predictions
    pred = Thresh(dfval[feat], mod, 0.5)
    # add to preprocess df
    dfval['buy'] = pred
    
    return dfval

