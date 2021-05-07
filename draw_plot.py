# -*- coding: utf-8 -*-

'''
    nc file read
    make map plot
    

'''

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
import pandas as pd
import datetime


# open nc file
url = '2007to2019.nc'
nc = Dataset(url, mode='r')

lat = nc['latitude'][:]
lon = nc['longitude'][:]
sst = nc['sst']

times = nc['time'][:]
units = nc['time'].units
dtime = num2date(times, units)


# time to dataframe file
df = pd.DataFrame(None, columns=['date'])
df['date'] = dtime
df['date_year'] = df['date'].dt.year


# time series graph for target area
# periods : 2016~2019
df_time = df.loc[df['date_year']>2015]
ls_index = df_time.index

sst_list = []
date_list = []
lat, lon = 26, 33

for l in ls_index:
    sst_list.append(sst[l, lat, lon]-273.15)
    date_list.append(df['date'].loc[l])
    
# draw time series graph
fig1 = plt.figure(figsize=(16, 10))
plt.rc('font', family='times new roman', size=20)
plt.title('SST of Pixel ['+str(lat)+']['+str(lon)+']', pad=20)
plt.plot(date_list, sst_list, color='black', label='SST(ºC)', linewidth=1)
plt.axhline(y=28, xmin=0, xmax=1)
plt.legend(loc='lower right')

x_ticks = []
for i in range(len(date_list)):
    if date_list[i].month==7 and date_list[i].day==1:
        x_ticks.append(date_list[i])

x_labels = np.arange(2016, 2020, 1)
plt.xticks(x_ticks, x_labels)

y_ticks = np.arange(10, 35, 5)
y_labels = np.arange(10, 35, 5)
plt.yticks(y_ticks, y_labels)
plt.show()
f_name = 'time series.png'
plt.savefig(f_name, bbox_inches='tight')
print('File Save: ' + f_name)



year = 2019
df_year = df.loc[df['date_year']==year]
ls_index = df_year.index

sst_list = []
date_list = []

for l in ls_index:
    sst_list.append(sst[l, lat, lon]-273.15)
    date_list.append(df['date'].loc[l])
    
# draw time series graph
fig2 = plt.figure(figsize=(16, 10))
plt.rc('font', family='times new roman', size=20)
plt.title('SST of Pixel ['+str(lat)+']['+str(lon)+']', pad=20)
plt.plot(date_list, sst_list, color='black', label='SST(ºC)', linewidth=1)
plt.axhline(y=28, xmin=0, xmax=1)
plt.legend(loc='lower right')
    
x_ticks = []
for i in range(len(date_list)):
    if date_list[i].day==15:
        x_ticks.append(date_list[i])

x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(x_ticks, x_labels)

y_ticks = np.arange(10, 35, 5)
y_labels = np.arange(10, 35, 5)
plt.yticks(y_ticks, y_labels)
plt.show()
f_name = 'time series '+str(year)+'.png'
plt.savefig(f_name, bbox_inches='tight')
print('File Save: ' + f_name)

    
    
    

# target area plot
fig3 = plt.figure(1, figsize=(10, 8))
ax = plt.gca()
plt.rc('font', family='times new roman', size=20)
plt.title('Study Area', weight='bold', ha='center', pad=10)

map = Basemap(projection='merc', resolution='h', 
                  urcrnrlat=np.nanmax(lat)+0.125, llcrnrlat=np.nanmin(lat)-0.125,
                  urcrnrlon=np.nanmax(lon)+0.125, llcrnrlon=np.nanmin(lon)-0.125)
    
    
map.drawcoastlines(linewidth=0.8)
map.fillcontinents(color='lightgrey')
parallels = np.arange(np.nanmin(lat)+1, np.nanmax(lat), 2)
map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.8,color='white')
merdians = np.arange(np.nanmin(lon)+3, np.nanmax(lon), 3)
map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.8,color='white')
map.drawmapboundary()

x,y = map(128.25, 34.5)
map.plot(x, y, 'ro', alpha=0.4 ,markersize=12)  

# x, y = map(127.3, 37)
# plt.text(x, y, 'KOREA', size=15)

plt.show()

f_name = 'study area.png'
plt.savefig(f_name, bbox_inches='tight')
print('File Save: ' + f_name)




# zoom in target area plot
# plt.figure(figsize=(10, 8))

# map = Basemap(projection='merc', resolution='h',
#               urcrnrlat=36, llcrnrlat=33, llcrnrlon=125, urcrnrlon=131)

# map.drawcoastlines(linewidth=0.8)
# map.fillcontinents(color='lightgrey')

# x,y = map(128.25, 34.5)
# map.plot(x, y, 'ro', alpha=0.4 ,markersize=25)  

# plt.show()

# f_name = 'study area zoom.png'
# plt.savefig(f_name, bbox_inches='tight')
# print('File Save: ' + f_name)