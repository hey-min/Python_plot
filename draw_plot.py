# -*- coding: utf-8 -*-
'''
draw plot

최종 수정일 : 2021.02.24 

'''

def drawPlot(data, date, text):
    ''' draw real Map '''
    
    plt.figure(figsize=(20, 16))
    ax = plt.gca()
    
    plt.rc('font', family='times new roman', size=30)
    
    plt.title(text, weight='bold', ha='center')
    
    map = Basemap(projection='cyl', resolution='h', 
                  urcrnrlat=np.nanmax(nc_lat), llcrnrlat=np.nanmin(nc_lat),
                  urcrnrlon=np.nanmax(nc_lon), llcrnrlon=np.nanmin(nc_lon))
    map.drawcoastlines(linewidth=3)
    map.drawlsmask(land_color='gray', ocean_color='white', lakes=True)
    parallels = np.arange(np.nanmin(nc_lat)+1, np.nanmax(nc_lat), 2)
    map.drawparallels(parallels, labels=[1,0,0,0], linewidth=0.5,color='white')
    merdians = np.arange(np.nanmin(nc_lon)+3, np.nanmax(nc_lon), 3)
    map.drawmeridians(merdians, labels=[0,0,0,1], linewidth=0.5,color='white')
    map.drawmapboundary()
    
    x,y = map(130, 40.5)
    plt.text(x, y, '('+str(date)+')', size=20)
    x,y = map(130, 40)
    avg_real = np.round(np.nanmean(plot_real), 2)
    avg_est = np.round(np.nanmean(plot_predict), 2)
    param_text_1 = 'real: '+str(avg_real)+' est: '+str(avg_est)
    plt.text(x, y, param_text_1, size=20)
    x,y = map(130, 39.5)
    param_text_2 = 'LR: '+str(lr)+' IT: '+str(it)
    plt.text(x, y, param_text_2, size=20)
    x,y = map(130, 39)
    param_text_3 = 'Train Years: '+str(train_data_size)+' Neurons: '+str(neurons_size)
    plt.text(x, y, param_text_3, size=20)
    
    
    x,y = map(nc_lon, nc_lat)
    
    min_sst = 20
    max_sst = 30
    
    levels = np.linspace(min_sst, max_sst, 50)
    img = plt.contourf(x, y, data, levels=levels, cmap='coolwarm', extend='both')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='3%', pad=0.5)
    
    cbar = plt.colorbar(img, ticks=levels, cax=cax)
    cbar.set_label(label='SST(℃)', labelpad=20)
    ticks = np.round(np.linspace(min_sst, max_sst, num=10, endpoint=True), 1)
    cbar.set_ticks(ticks)
    
    plt.savefig(PATH_OF_IMG + model_name + '.png', bbox_inches='tight')
    print('File Save: ' + PATH_OF_IMG + model_name + '.png')
    plt.show()