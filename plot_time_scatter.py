# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:12:44 2021

@author: Kiost


Input : lat, lon

Output
1. 연도별 수온 시계열 그래프
2. 예측한 시계열 그래프
3. 산점도 다이어그램 그림

"""

from netCDF4 import Dataset, num2date

file = 'sst_dataset_2007to2020.nc'
from netCDF4 import Dataset, num2date
nc = Dataset(file)
    
var = nc.variables.keys()
print(var)
    
lat = nc.variables['longitude'][:]
lon = nc.variables['latitude'][:]
nctime = nc.variables['time'][:] # get values
t_unit = nc.variables['time'].units # get unit  "days since 1950-01-01T00:00:00Z"
t_cal = nc.variables['time'].calendar
nc_time = num2date(nctime,units = t_unit,calendar = t_cal)
sst = nc.variables['sst'][:]

import pandas as pd
import datetime as dt
import cftime


df_time = pd.DataFrame(None, columns=['date', 'date_year'])
df_time['date'] = nc_time

for i in range(len(nc_time)):
    df_time['date_year'][i] = df_time['date'][i].year
    
date_year = df_time.loc[df_time['date_year']==2019]   
date_total = df_time





from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

lat, lon = 27, 33

EST = 1
YEAR = 365
SIZE_BATCH = 30 
N_STEPS = 30

# start = int(min(df_time_input.index))
# end = int(max(df_time_input.index))

# print('Input : {} ~ {}' .format(min(df_time_input['date']), max(df_time_input['date'])))
# find_year_cnt = df_time_input['date'].count()


# [lat][lon]의 SST TOTAL과 YEAR 생성
def get_data(lat, lon, year):
    
    sst_total = []
    sst_year = []
    date_total = []
    # date_year = []
    
    array_total_layer = np.array(df_time.index)
    
    for l in array_total_layer:
        
        data = np.round(sst[l, lat, lon] - 273.15, 4)
        
        #if find year
        if df_time.loc[l].date_year == year:
            sst_year.append(data)
        
            # date_year.append(df_time.loc[l].date)
        sst_total.append(data)
        # date_total.append(df_time.loc[l].date)
    
    return sst_total, sst_year, date_year

find_year = 2019
sst_total, sst_year, date_year = get_data(lat, lon, find_year)
# print('> generate dataframe')


    
    
    
def plot_timeSeries_year(lat, lon): 
    
    
    plt.figure(figsize=(15, 12))
    ax = plt.gca()
        
    plt.rc('font', size=25)

    title = 'Time Series Graph for 2019 (Pixel ['+str(lat)+']['+str(lon)+'] )'
    plt.title(title, fontdict={'fontsize':30}, loc='left', pad=20)
    
    plt.plot(date_year.index, sst_year, label='sea surface temperatures')
    plt.legend(loc='lower right')
    plt.xlabel('Days', fontsize=20, labelpad=15)
    plt.ylabel("SST(℃)", fontsize=20, labelpad=15)
    plt.axhline(y=28, color='r', linewidth=1) 
    plt.grid()
    
    x_ticks = []
    for i in date_year.index:
        if date_year.loc[i].date.day==15:
            x_ticks.append(i)

    x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.xticks(x_ticks, x_labels)

    y_ticks = np.arange(10, 35, 5)
    y_labels = np.arange(10, 35, 5)
    

    path = 'ts_'+str(find_year)+'_'+str(lat)+'_'+str(lon)+'.png'
    plt.savefig(path, bbox_inches='tight')
    print('File Save: {}' .format(path))
    
    plt.show()

# print('> plot_timeSeries_year')
# plot_timeSeries_year(lat, lon)





def plot_timeSeries_total(sst_list): 
    
    plt.figure(figsize=(15, 12))
    ax = plt.gca()
        
    plt.rc('font', size=25)

    title = 'Time Series Graph for (Pixel ['+str(lat)+']['+str(lon)+'] )'
    plt.title(title, fontdict={'fontsize':30}, loc='left', pad=20)
    
    plt.plot(date_total.index, sst_list, label='sea surface temperatures')
    plt.legend(loc='lower right')
    plt.xlabel('Days', fontsize=20, labelpad=15)
    plt.ylabel("SST(℃)", fontsize=20, labelpad=15)
    plt.axhline(y=28, color='r', linewidth=1) 
    plt.grid()
    
    x_ticks = []
    x_labels = np.arange(2007, 2020, 2)
    
    for i in date_total.index:
        data = date_total.loc[i].date
        
        if data.year in x_labels:
            if data.month==7 and data.day==15:
                x_ticks.append(i)
                print(i)

    
    plt.xticks(x_ticks, x_labels)

    y_ticks = np.arange(10, 35, 5)
    y_labels = np.arange(10, 35, 5)
    plt.show()
    
    path = 'ts_total_'+str(lat)+'_'+str(lon)+'.png'
    plt.savefig(path, bbox_inches='tight')
    print('File Save: {}' .format(path))
    
    plt.show()

print('> plot_timeSeries_total')
plot_timeSeries_total(sst_total)







# time_est = np.zeros(find_year_cnt)
# time_label = np.zeros(find_year_cnt)

# est = np.zeros(find_year_cnt-EST)
# label = np.zeros(find_year_cnt-EST)



'''
def plot_scatter():
    
    x = np.array(rslt_real)
    y = np.array(rslt_est)
    
    denominator = x.dot(x) - x.mean() * x.sum()
    m = ( x.dot(y) - y.mean() * x.sum() ) / denominator
    b = ( y.mean() * x.dot(x) - x.mean() * x.dot(y) ) / denominator
    y_pred = m*x + b
    
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
        
    plt.rc('font', size=25)
    
    title = 'Scatter Diagram for 2019 (Pixel ['+str(LAT)+']['+str(LON)+'] )'
    plt.title(title, fontdict={'fontsize':30}, loc='left', pad=20)
    
    plt.xlim([5, 30])
    plt.ylim([5, 30])
    plt.xticks([10,15,20,25])
    plt.yticks([10,15,20,25])
    
    plt.scatter(rslt_real, rslt_est, c='red', s=5)
    plt.rcParams['lines.linewidth'] = 1
    plt.plot(x, y_pred, 'blue')
    
    plt.xlabel('Real SST', labelpad=20)
    
    if EST==1:
        plt.ylabel('Estimated SST('+str(EST)+'-day)', labelpad=20)
    else:
        plt.ylabel('Estimated SST('+str(EST)+'-days)', labelpad=20)
        
    text = 'R$^2$: ' + str(r2) + '  RMSE: '+ str(rmse) +'  MAPE: ' + str(mape)
    text2 = '\nLR: ' + str(LR) + '  IT: ' + str(IT)
    textbox = offsetbox.AnchoredText(text+text2, loc='upper left')
    ax.add_artist(textbox)
    
    #### SAVE IMAGE ####
    img_file = 'scatter_'+save_name+'.png'
    plt.savefig(img_file, bbox_inches='tight')
    print('File Save: {}' .format(img_file))
    
    plt.show()    
    


def plot_timeSeries_with_est(lat, lon): 

    plt.figure(figsize=(15, 12))
    ax = plt.gca()
        
    plt.rc('font', size=25)

    title = 'Time Series Graph for 2019 (Pixel ['+str(lat)+']['+str(lon)+'] )'
    plt.title(title, fontdict={'fontsize':30}, loc='left', pad=20)
    
    plt.plot(rslt_est, label='EST-'+str(EST)+'Day(s) SST', color='blue')
    plt.plot(rslt_real, label='Label', color='gray')
    plt.legend(loc='lower right')
    plt.xlabel('Days')
    plt.ylabel("SST(℃)")
    plt.axhline(y=28, color='r', linewidth=1) 
    plt.grid()
    plt.yticks(np.arange(10, 35, 5))
    
    
    #### SAVE IMAGE ####
    img_file = 'timeSeries_'+save_name+'.png'
    plt.savefig(img_file, bbox_inches='tight')
    print('File Save: {}' .format(img_file))
    
    plt.show()


def accuracy():
    
    df_real = rslt_real
    df_est = rslt_est
    
    for i in range(len(df_real)):
    
        real = df_real[i]
        est = df_est[i]
        df_real[i] = round(real, 4)
        df_est[i] = round(est, 4)


    y_true = df_real
    y_pred = df_est

    # R2, RMSE, MAPE
    R2 = 0
    RMSEa = 0
    MAPEa = 0
    k = 0
    for k in range(len(df_real)):
        
        Xlist = y_true
        Ylist = y_pred
        x = np.array(y_true)
        y = np.array(y_pred)
   
        res = x - y
        tot = x - y.mean()   
        R2 = 1 - np.sum(res**2) / np.sum(tot**2)
        
        RMSE = np.sqrt(((y-x) ** 2).mean())
    
        MAPE = np.mean(np.abs((x - y)/x)) * 100

    columns = ['EST', 'R2', 'RMSE', 'MAPE']

    data_ac = {'EST':[EST],
        'R2':[round(R2,4)],
        'RMSE':[round(RMSE,2)],
        'MAPE':[round(MAPE,2)]}
    
    df_ac = pd.DataFrame(data_ac, index=data_ac['EST'])
    
    # TP, FP, FN, TN
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for i in range(len(df_real)):
        if y_true[i] > 28 and y_pred[i] > 28:
            TP = TP + 1
        if y_true[i] < 28 and y_pred[i]> 28:
            FP = FP + 1
        if y_true[i] > 28 and y_pred[i] < 28:
            FN = FN + 1
        if y_true[i] < 28 and y_pred[i] < 28:
            TN = TN + 1


    # precision, recall, accuracy, f1 score
    TPR = 0
    FPR = 0
    Precision = 0
    Recall = 0
    F1 = 0
    Accuracy = 0
    
    
    for i in range(len(df_real)):
        
        if TP == 0: break
        TPR = TP / (TP+FN)
        FPR = FP / (FP+TN)
        
        Precision = TP / (TP+FP)
        Recall = TP / (TP+FN)
        F1 = 2*(Precision * Recall) / (Precision + Recall)
        Accuracy = (TP + TN) / (TP + FN + FP + TN)


    columns = ['EST', 'accuracy', 'recall', 'precision', 'f1_score', 'TPR', 'FPR']

    data_hwt = {'EST':[EST], 'accuracy':[round(Accuracy,2)],
        'recall':[round(Recall,2)],
        'precision':[round(Precision,2)],
        'f1_score':[round(F1,5)],
        'TPR':[round(TPR,5)],
        'FPR':[round(FPR,5)]}

    df_hwt = pd.DataFrame(data_hwt, index=data_hwt['EST'])
    
    df_rslt = pd.merge(df_ac, df_hwt, on='EST')
    
    
    return df_rslt

'''