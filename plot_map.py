# -*- coding: utf-8 -*-

# nc file 읽고 Map 생성 

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, num2date
import pandas as pd
import datetime


# 1) open nc file
# url = '2007to2019.nc'
# nc = Dataset(url, mode='r')

# lat = nc['latitude'][:]
# lon = nc['longitude'][:]
# sst = nc['sst']

# times = nc['time'][:]
# units = nc['time'].units
# dtime = num2date(times, units)


# # time to dataframe file
# df = pd.DataFrame(None, columns=['date'])
# df['date'] = dtime
# df['date'] = pd.to_datetime(df['date'])
# df['date_year'] = df['date'].dt.year


# 2) Open version2 nc file

list = [[3, 55], [3, 40]]

_lat, _lon = 26, 35

from netCDF4 import Dataset, num2date

file = '2007to2019_sst_temp_press.pickle.nc'
nc = Dataset(file)

nc_lat = nc.variables['latitude'][:]
nc_lon = nc.variables['longitude'][:]
nctime = nc.variables['time'][:]
t_unit = nc.variables['time'].units
t_cal = nc.variables['time'].calendar
nc_time = num2date(nctime,units = t_unit,calendar = t_cal)
nc_sst = nc.variables['sst'][:]
nc_skt = nc.variables['skt'][:]
nc_sp = nc.variables['sp'][:]

find_date = datetime.datetime(2019,8,10,12,0)
_id = np.where(nc_time==find_date)[0][0]

df1 = pd.DataFrame(nc_sst[_id]-273.15).to_numpy()
df2 = pd.DataFrame(nc_skt[_id]-273.15).to_numpy()
df3 = pd.DataFrame(nc_sp[_id]).to_numpy()

def draw_map_SST(title_type='', df=''):
    
    import matplotlib.pyplot as plt
    import matplotlib.offsetbox as offsetbox
    from mpl_toolkits.basemap import Basemap
    from mpl_toolkits.axes_grid1 import make_axes_locatable


    # x = df.flatten()
    # x = x[~np.isnan(x)]
    
    plt.figure(figsize=(20, 16))
    ax = plt.gca()

    plt.rc('font', size=25)
    
    month_str = find_date.month
    plt.title(title_type, loc='left', pad=20)
    plt.title('10 Aug 2019 12:00', loc='right', pad=20)

    map = Basemap(projection='merc', resolution='h', 
                  urcrnrlat=np.nanmax(nc_lat)+0.125, llcrnrlat=np.nanmin(nc_lat)-0.125,
                  urcrnrlon=np.nanmax(nc_lon)+0.125, llcrnrlon=np.nanmin(nc_lon)-0.125)

    map.drawcoastlines(linewidth=0.8)
    
    map.fillcontinents(color='lightgrey')
    
    parallels = np.arange(np.nanmin(nc_lat)+1, np.nanmax(nc_lat), 2)
    
    map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.8,color='white')
    
    merdians = np.arange(np.nanmin(nc_lon)+3, np.nanmax(nc_lon), 3)
    map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.8,color='white')
    
    map.drawmapboundary()
    
    

    # tb_data = 'Real Average '+title_type
    # text_left = offsetbox.AnchoredText(tb_data, loc='upper left')
    # ax.add_artist(text_left)
    
    
    lons, lats = np.meshgrid(nc_lon, nc_lat)
    x,y = map(lons, lats)
    
    cmap = plt.get_cmap('jet')
    _vmin = 20
    _vmax = 30
    levels = np.arange(_vmin, _vmax+1, 2)
        
    img = map.pcolormesh(x, y, df, cmap=cmap, shading='gouraud')
    img.set_clim(vmin=_vmin, vmax=_vmax)
    
    divider = make_axes_locatable(ax)
    
    # 하단 colorbar 배치
    cax = divider.append_axes('bottom', size='5%', pad='10%')
    cbar = plt.colorbar(img, ticks=levels, cax=cax, orientation='horizontal', extend='both')
    cbar.set_label(label='Temperature[℃]', labelpad=10)
    # cbar.set_label(label='Pressure [Pa]', labelpad=10)

    plt.show()


def draw_map_sp(df=''):
    
    import matplotlib.pyplot as plt
    import matplotlib.offsetbox as offsetbox
    from mpl_toolkits.basemap import Basemap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.figure(figsize=(20, 16))
    ax = plt.gca()

    plt.rc('font', size=25)
    
    plt.title('Surface Pressure', loc='left', pad=20)
    plt.title('10 Aug 2019 12:00', loc='right', pad=20)

    map = Basemap(projection='merc', resolution='h', 
                  urcrnrlat=np.nanmax(nc_lat)+0.125, llcrnrlat=np.nanmin(nc_lat)-0.125,
                  urcrnrlon=np.nanmax(nc_lon)+0.125, llcrnrlon=np.nanmin(nc_lon)-0.125)

    map.drawcoastlines(linewidth=0.8)
    
    map.fillcontinents(color='lightgrey')
    
    parallels = np.arange(np.nanmin(nc_lat)+1, np.nanmax(nc_lat), 2)
    
    map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.8,color='white')
    
    merdians = np.arange(np.nanmin(nc_lon)+3, np.nanmax(nc_lon), 3)
    map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.8,color='white')
    
    map.drawmapboundary()
    

    
    lons, lats = np.meshgrid(nc_lon, nc_lat)
    x,y = map(lons, lats)
    
    cmap = plt.get_cmap('jet')
    map.pcolormesh(x, y, df, cmap=cmap, shading='gouraud')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('bottom', size='5%', pad='10%')
    
    cbar = plt.colorbar(cax=cax, orientation='horizontal', extend='both')

    cbar.set_label(label='Pressure [Pa]', labelpad=10)

    plt.show()


# draw_map(title_type='Sea Surface Temperautre', df=df1)
# draw_map(title_type='Skin Temperautre', df=df2)

# draw_map_sp(df=df3)





def draw_time(): 
    
    # time series graph for target area
    
    find_date = datetime.datetime(2019,1,1,12,0)
    _id = np.where(nc_time==find_date)[0][0]
    
    sst_list = nc_sst[:_id, _lat, _lon]-273.15
    # skt_list = nc_skt[:, _lat, _lon]-273.15
    # sp_list = nc_sp[:, _lat, _lon]/5000
    date_list = nc_time[:_id]
    

 
    # draw time series graph
    fig1 = plt.figure(figsize=(25, 10))
    ax = plt.gca()
    plt.rc('font', size=25)
    # plt.title('Model Total Data', loc='left', pad=20)
    plt.title('Sea Surface Temperature', loc='left', pad=20)
    plt.title('Lat: '+str(nc_lat[_lat])+' Lon: '+str(nc_lon[_lon]), loc='right', pad=20)
    
    plt.plot(date_list, sst_list,  label='SST [℃]', linewidth=1)
    # plt.plot(date_list, skt_list,  label='SKT[℃]', linewidth=1)
    # plt.plot(date_list, sp_list,  label='SP[Pa]/5000', linewidth=1)
    plt.axhline(y=28, xmin=0, xmax=1, color='red')
    plt.legend(loc='upper left')
    
    # x_ticks = []
    # for i in range(len(date_list)):
    #     if date_list[i].month==7 and date_list[i].day==1:
    #         x_ticks.append(date_list[i])

    # x_labels = np.arange(2016, 2020, 1)
    # plt.xticks(x_ticks, x_labels)
    
    y_ticks = np.arange(10, 35, 5)
    y_labels = np.arange(10, 35, 5)
    plt.yticks(y_ticks, y_labels)
    
    plt.ylabel('Temperature [℃]', labelpad=20)   
    plt.xlabel('Year', labelpad=20)
    plt.grid(True)
    
    # f_name = 'time series.png'
    # plt.savefig(f_name, bbox_inches='tight')
    # print('File Save: ' + f_name)
    
    plt.show()

# draw_time()

def draw_time_year(year=2018):
    
    
    find_date = datetime.datetime(year,1,1,12,0)
    _id = np.where(nc_time==find_date)[0][0]
    
    sst_list = nc_sst[_id:_id+366, _lat, _lon]-273.15
    date_list = nc_time[_id:_id+366]
    
    
    # draw time series graph
    fig2 = plt.figure(figsize=(16, 10))
    ax = plt.gca()
    plt.rc('font', size=25)
    plt.title('Sea Surface Temperautre in '+str(year), loc='left', pad=20)
    plt.title('Lat: '+str(nc_lat[_lat])+' Lon: '+str(nc_lon[_lon]), loc='right', pad=20)
    
    plt.plot(date_list, sst_list, label='SST [℃]', linewidth=1)
    plt.axhline(y=28, xmin=0, xmax=1, color='red')
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
    
    plt.ylabel('Temperature [℃]', labelpad=20)   
    plt.xlabel('Year', labelpad=20)
    plt.grid(True)
    
    # f_name = 'time series '+str(year)+'.png'
    # plt.savefig(f_name, bbox_inches='tight')
    # print('File Save: ' + f_name)
    
    plt.show()

# draw_time_year(2018)    

def draw_time_window(year=2018):
    
    
    day_first = datetime.datetime(year,8,1,12,0)
    _start = np.where(nc_time==day_first)[0][0]
    day_last = datetime.datetime(year,8,15,12,0)
    _last = np.where(nc_time==day_last)[0][0]
    
    day_est = datetime.datetime(year,8,16,12,0)
    _est = np.where(nc_time==day_est)[0][0]
    
    
    sst_list = nc_sst[_start:_est+1, _lat, _lon]-273.15
    date_list = nc_time[_start:_est+1]
    x_ticks = np.arange(1, 17, 1)
    
    # draw time series graph
    fig2 = plt.figure(figsize=(16, 10))
    ax = plt.gca()
    plt.rc('font', size=25)
    plt.title(' ' + day_first.strftime('%b')+' '+str(year), loc='left', pad=20)
    plt.title('Lat: '+str(nc_lat[_lat])+' Lon: '+str(nc_lon[_lon]), loc='right', pad=20)
    
    plt.plot(x_ticks, sst_list)
    plt.axhline(y=28, xmin=0, xmax=1, color='red')
    # plt.legend(loc='lower right')
        
    
    plt.xticks(x_ticks)
    
    y_ticks = np.arange(27.5, 29.5, 0.5)
    y_labels = np.arange(27.5, 29.5, 0.5)
    plt.yticks(y_ticks, y_labels)
    
    plt.ylabel('Temperature [℃]', labelpad=20)   
    plt.xlabel('Day', labelpad=20)
    plt.grid(True)
    
    # f_name = 'time series '+str(year)+'.png'
    # plt.savefig(f_name, bbox_inches='tight')
    # print('File Save: ' + f_name)
    
    plt.show()

draw_time_window(2018)

def draw_target():
    
    # target area plot
    fig3 = plt.figure(figsize=(16, 10))
    ax = plt.gca()
    plt.rc('font', size=25)
    plt.title('Study Area', ha='center', pad=10)
        
    map = Basemap(projection='merc', resolution='h', 
                      urcrnrlat=np.nanmax(nc_lat)+0.125, llcrnrlat=np.nanmin(nc_lat)-0.125,
                      urcrnrlon=np.nanmax(nc_lon)+0.125, llcrnrlon=np.nanmin(nc_lon)-0.125)
        
        
    map.drawcoastlines(linewidth=0.8)
    map.fillcontinents(color='lightgrey')
    parallels = np.arange(np.nanmin(nc_lat)+1, np.nanmax(nc_lat), 2)
    map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.8,color='white')
    merdians = np.arange(np.nanmin(nc_lon)+3, np.nanmax(nc_lon), 3)
    map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.8,color='white')
    map.drawmapboundary()
    
    for i in range(len(list)):
        r, c = list[i][0], list[i][1]
        x,y = map(nc_lon[c], nc_lat[r])
        map.plot(x, y, 'ro', alpha=0.4 ,markersize=15)
        
        x,y = map(nc_lon[c]-0.3, nc_lat[r]-0.3)
        plt.text(x, y, str(i+1), size=30)
    
    
    x, y = map(127.3, 37)
    plt.text(x, y, 'KOREA', size=15)
    
    # f_name = 'study area.png'
    # plt.savefig(f_name, bbox_inches='tight')
    # print('File Save: ' + f_name)
    
    plt.show()


def draw_target_zoom(_lat, _lon):
    
    # zoom in target area plot
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.rc('font', size=25)
    map = Basemap(projection='merc', resolution='h',
                  urcrnrlat=36, llcrnrlat=33, llcrnrlon=125, urcrnrlon=131)
    
    map.drawcoastlines(linewidth=0.8)
    map.fillcontinents(color='lightgrey')
        
    x,y = map(nc_lon[_lon], nc_lat[_lat])
    map.plot(x, y, 'ro', alpha=0.4 ,markersize=25)  
        
    # f_name = 'study area zoom.png'
    # plt.savefig(f_name, bbox_inches='tight')
    # print('File Save: ' + f_name)
    
    plt.show()

    
  