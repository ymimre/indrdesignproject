# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 16:07:34 2021

@author: ymert
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from datetime import datetime, timedelta
import math
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statistics

warnings.filterwarnings("ignore")
#Import Data
sales_data = pd.read_csv("sales.csv",usecols = ['Magaza_Kodu','Satis_Tarihi','Urunkodu','Satis_Adet'])
store_data = pd.read_csv("Store_Master.csv")
product_data = pd.read_csv("Product_Master.csv",usecols = ['UrunKodu','AileKodu'])

#Drop Unwanted Store Data
store_data = store_data[store_data.REGION == 35]
store_data = store_data[store_data.Magaza_Kapanis_Tarihi== 0]
store_data = store_data[store_data.Magaza_Acilis_Tarihi<= 20191105]
#Put Product Codes and Store Codes Into an Array
product_codes = product_data['UrunKodu'].to_numpy()
store_codes = store_data['Magaza Kodu'].to_numpy()

#Grouping products with respect to their sale volumes
volume = []
total = 0
sales_data.set_index('Satis_Tarihi', inplace=True)
sales_data.index = pd.to_datetime(sales_data.index)
date_index = pd.date_range('11/05/2019', '11/03/2021', freq='D')
d = pd.DataFrame(np.zeros([730,2]), columns=['Urunkodu','Satis_Adet'])
d.index=date_index
for j in range(len(product_codes)):
    df = sales_data[sales_data.Urunkodu == product_codes[j]]
    summed = df['Satis_Adet']
    if len(summed) >0:
        summed = summed.resample('3M').sum()
        if(summed.iloc[-1]>0):
            for i in range(len(store_codes)):
                df1 = df[df.Magaza_Kodu ==str(store_codes[i])]
                df1 = df1.append(d, ignore_index=False)
                df1 = df1.resample('1D').sum()
                temp = df1['Satis_Adet'].sum()
                summed2 = df1['Satis_Adet']
                if len(summed2) >0:
                    summed2 = summed2.resample('3M').sum()
                   
                    if(summed2.iloc[-1]>0):
                        if temp > 0:
                            total = total + temp;
                            volume.append([temp,store_codes[i],product_codes[j]])
                    
volume.sort(reverse = True)
high_volume = volume[:math.floor(len(volume)*0.2)]

"""
unique = []
for x in [row[2] for row in high_volume]:
    if x not in unique:
        unique.append(x)

stores_unique = []
for x in unique:
    stores_unique_temp = product_data[product_data.UrunKodu == x]
    stores_unique_temp2 = int(stores_unique_temp['AileKodu'])
    if stores_unique_temp2 not in stores_unique:
        stores_unique.append(stores_unique_temp2)
"""

#Forecast
total_mse_ma = 0
total_mse_nots = 0
total_mse_ts = 0
mse = []
forecast = []
summed_y =0 
for i in range(len(high_volume)):
    df2 = sales_data[sales_data.Urunkodu == high_volume[i][2]]
    df3 = df2[df2.Magaza_Kodu ==high_volume[i][1]]
    daily = df3.resample('1D').sum()
    y = daily['Satis_Adet']
    summed_y = summed_y + sum(y)
    y_train = y.iloc[:-14]
    Q1 = int(y_train.quantile([0.25]))
    Q3 = int(y_train.quantile([0.75]))
    minimum = Q1 -1.5*(Q3-Q1)
    maximum = Q3 +1.5*(Q3-Q1)
    temp = y_train.mean()+y_train.std()
    temp2 = y_train.mean()-y_train.std()
    for j in range(7,len(y_train)):
        if y_train[j] > maximum:
            y_train[j] = Q3
        if y_train[j] < minimum:
            y_train[j] =Q1
    y_test = y.iloc[-14:]
    #Moving Averages
    y_pred_ma = y.rolling(8).mean() 
    y_pred_ma = y_pred_ma.iloc[-14:]
    y_pred_ma = (y_pred_ma*8 - y_test)/7 
    meanSqErr_ma = metrics.mean_squared_error(y_test, y_pred_ma)
    total_mse_ma = total_mse_ma + meanSqErr_ma
    #Holt-Winters with trend and seasonality
    model_holt_ts = ExponentialSmoothing(endog = y_train,
                             trend = "add",
                             seasonal = "add",
                             seasonal_periods = 7).fit()
    y_pred_holt_ts = model_holt_ts.forecast(steps = 14)
    meanSqErr_holt_ts = metrics.mean_squared_error(y_test, y_pred_holt_ts) 
    total_mse_ts = total_mse_ts + meanSqErr_holt_ts
    #Holt-Winters without trend and seasonality
    model_holt_nots = ExponentialSmoothing(endog = y_train).fit()
    y_pred_holt_nots = model_holt_nots.forecast(steps = 14)
    meanSqErr_holt_nots = metrics.mean_squared_error(y_test, y_pred_holt_nots) 
    total_mse_nots = total_mse_nots + meanSqErr_holt_nots
    mse.append([meanSqErr_ma,meanSqErr_holt_ts,meanSqErr_holt_nots,high_volume[i][1],high_volume[i][2]])
    
    minimum = min(meanSqErr_ma, meanSqErr_holt_ts, total_mse_nots)
    if minimum == meanSqErr_ma:
        y_pred_final = []
        y_2 = y.iloc[-7:]
        for i in range(14):
            temp3 = y_2.mean()
            y_pred_final.append(temp3)
            y_2[i % 7 ] = temp3
    elif minimum == meanSqErr_holt_ts:
        model_holt_ts = ExponentialSmoothing(endog = y,
                             trend = "add",
                             seasonal = "add",
                             seasonal_periods = 7).fit()
        y_pred_final = model_holt_ts.forecast(steps = 14)
    else:
        model_holt_nots = ExponentialSmoothing(endog = y).fit()
        y_pred_final = model_holt_nots.forecast(steps = 14)
    forecast.append([y_pred_final,high_volume[i][1],high_volume[i][2]])
    """
    #ARIMA
    model_arima = SARIMAX(y_train, 
                order = (3, 0, 2), 
                seasonal_order =(2, 1, 0, 12))
    result = model_arima.fit()
    start = len(y_train)
    end = len(y_train) + len(y_test) - 1
    y_pred_arima = result.predict(start, end,
                             typ = 'levels')
    meanSqErr_arima = metrics.mean_squared_error(y_test, y_pred_arima)
    """
    if i == 1:
        plt.plot(y_test, label = 'Test values')
        plt.plot(y_pred_ma, label = 'Moving Average')
        plt.plot(y_pred_holt_ts, label = 'Holt with trend and seasonality')
        plt.plot(y_pred_holt_nots, label = 'Holt without trend and seasonality')
        #plt.plot(y_pred_arima, label = 'Arima')
        plt.legend()
