# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:20:25 2021

@author: Kiost

# 고수온 카운트해서 각 연도별로 PLOT 생성

"""

import numpy as np
import os
import math
import datetime
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd


def read_pickle():

    with open('2007to2019.pickle', 'rb') as f:
        nc_sst = pickle.load(f)
        nc_lat = pickle.load(f)
        nc_lon = pickle.load(f)
        nc_time = pickle.load(f)

    return nc_sst, nc_lat, nc_lon, nc_time 


nc_sst, nc_lat, nc_lon, nc_time  = read_pickle()

years = np.arange(2007, 2020, 1)
df_year = pd.DataFrame(None, columns=['year', 'start', 'end'], index=None)

def get_sst(year):
    
    global df_year
    
    start = datetime.datetime(year, 1, 1, 12, 0)
    end = datetime.datetime(year, 12, 31, 12, 0)
    
    days = nc_time.tolist()
    
    id_start = days.index(start)
    id_end = days.index(end)
    
    new = {'year':year, 'start':id_start, 'end':id_end}
    df_year = df_year.append(new, ignore_index=True)
    print('{}: {}~{}' .format(year, id_start, id_end))
    
    return nc_sst[id_start:id_end+1]

def count_hwt(sst):

    cnt = [[0 for c in range(len(nc_lon))] for r in range(len(nc_lat))]   
    for l in range(len(sst)):
        for r in range(len(nc_lat)):
            for c in range(len(nc_lon)):
                pixel = sst[l, r, c]
            
                if math.isnan(pixel):
                    cnt[r][c] = None
                    continue
            
                data = pixel - 273.15
            
                if data >= 28:
                    cnt[r][c] = cnt[r][c] + 1
    
    rslt = pd.DataFrame(cnt)
    return rslt

def draw(hwt_map, year):
    
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    plt.rc('font', family='times new roman', size=20)
    
    title = 'Counts of HWT in ' + str(year)
    plt.title(title, weight='bold', ha='center', pad=10)
    
    map = Basemap(projection='merc', resolution='h', 
                  urcrnrlat=np.nanmax(nc_lat)+0.125, llcrnrlat=np.nanmin(nc_lat)-0.125,
                  urcrnrlon=np.nanmax(nc_lon)+0.125, llcrnrlon=np.nanmin(nc_lon)-0.125)
    
    
    map.drawcoastlines(linewidth=0.8)
    
    map.fillcontinents(color='lightgrey')
    
    
    parallels = np.arange(np.nanmin(nc_lat)+1, np.nanmax(nc_lat), 2)
    map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.5,color='white')
    
    merdians = np.arange(np.nanmin(nc_lon)+3, np.nanmax(nc_lon), 3)
    map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.5,color='white')
    
    map.drawmapboundary()
    
    lons, lats = np.meshgrid(nc_lon, nc_lat)
    x,y = map(lons, lats)
    
    min_sst = 0
    max_sst = 100
    
    levels = np.linspace(min_sst, max_sst, 100)
    cmap = plt.get_cmap('coolwarm')
    map.pcolormesh(x, y, hwt_map, cmap=cmap, shading='gouraud')
    img = plt.contourf(x, y, hwt_map, levels=levels, cmap='jet')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.2)
    
    cbar = plt.colorbar(img, ticks=levels, cax=cax)
    # cbar.set_label(label='Counts', labelpad=10)
    ticks = np.arange(0, 100, 10)
    cbar.set_ticks(ticks)
    
    
    
    path = 'img/hwt_'+str(year)+'.png'
    plt.savefig(path, bbox_inches='tight')
    print('File Save: ' + path)
    plt.show()

    
ssts = pd.DataFrame(None)
for year in years:
    
    # year의 sst df에서 고수온이면 새로 df 만들어서 draw
    sst_of_year = get_sst(year)
    map_of_hwt = count_hwt(sst_of_year)
    
    draw(map_of_hwt, year)
    
            
    

        