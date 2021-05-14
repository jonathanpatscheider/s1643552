#!/usr/bin/env python
# coding: utf-8

# In[65]:


from __future__ import (absolute_import, division, print_function)
import json
import pandas as pd
import requests
import numpy as np
import html
import folium
import copy
from folium import plugins
from folium.plugins import HeatMap, TimeSliderChoropleth, MarkerCluster, HeatMapWithTime
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.io import output_file, show
from bokeh.palettes import GnBu3, OrRd3, Category20,Category10, Spectral6
from bokeh.plotting import figure, output_file, save
from bokeh.transform import factor_cmap
from collections import Counter
import googlemaps
from datetime import datetime
import geopandas as gpd
import os
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as offline
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from shapely.geometry import Point, Polygon
from geopandas import GeoSeries, GeoDataFrame
import contextily as ctx
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widget
get_ipython().run_line_magic('matplotlib', 'inline')

import requests
import json
import pandas as pd
%matplotlib inline
import plotly.express as px

from __future__ import (absolute_import, division, print_function)
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import folium
from folium.plugins import TimeSliderChoropleth
import numpy as np

from shapely.geometry import Point
from shapely.geometry import Polygon
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame
# ## Data preparation 
# Since some categories are present multiple times but with different spelling. The found categories are:
# - "Illness", "illnes" and "Illnes" 
# - "Drugs (illegal)" and "Drugs (Illegal)"
# - "Cell Phone (hand-held)" "Cell Phone (hand-Held)"
# - "Reaction to Uninvolved Vehicle" and "Reaction to Other Uninvolved Vehicle"
# These duplicates are merged and in the further analysis called by the first name mentioned in the list above.
# 
# The same goes for vehicle types where:
# - "E-Bike","E-Bik", "BICYCLE", "Pedicab" and "Bike are all treated in the same category "Bike"
# - "Motorbike", "MOTORCYCLE" and "Motorcycle" are all treated in the same category "Motorcycle"
# - "Taxi" and "TAXI" are handled as "TAXI"
# - "VAN" and "Van" are handled as "Van"
# - "E-Sco" and "E-Scooter" are handled as "E-Scooter"
for j in range(18,22+1):
    for i in range(0,len(data)):
        lookup = data.iloc[i,j]
        if lookup == "illnes" or lookup == "Illnes":
            data.iloc[i,j] = "Illness"
        elif lookup == "Drugs (Illegal)":
            data.iloc[i,j] = "Drugs (illegal)"
        elif lookup == "Cell Phone (hand-Held)":
            data.iloc[i,j] = "Cell Phone (hand-held)"
        elif lookup == "Reaction to Other Uninvolved Vehicle":
            data.iloc[i,j] = "Reaction to Uninvolved Vehicle"
        else:
            passfor j in range(-6,-1):
    for i in range(0,len(data)):
        lookup = data.iloc[i,j]
        if lookup == "E-Bike" or lookup == "BICYCLE" or lookup == "E-Bik" or lookup == "Pedicab":
            data.iloc[i,j] = "Bike"
        elif lookup == "Motorbike" or lookup == "MOTORCYCLE":
            data.iloc[i,j] = "Motorcycle"
        elif lookup == "Taxi":
            data.iloc[i,j] = "TAXI"
        elif lookup == "VAN":
            data.iloc[i,j] = "Van"
        elif lookup == "E-Sco":
            data.iloc[i,j] = "E-Scooter"
        else:
            pass
# As these steps takes a long time to run the dataframe is saved in a csv.
data.to_csv('data_preparred.csv', index=False)
# #### Preparring pedestrian and cyclist dataframes 
# Secondly a dataframe for the accidents involving pedestrian and a dataframe for the accidents involving bicyclists in the period from 2016 to 2020 are generated. 
data['datetime']=pd.to_datetime(data['CRASH DATE'] + ' ' + data['CRASH TIME'])
filterdf = data[(pd.to_datetime(data['datetime']) >= '01/01/2016') & (pd.to_datetime(data['datetime']) <= '31/12/2020')]

index_pedestrian = []
index_cyclist = []
textt = 'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion'
for i in range(0,len(filterdf)):
    Bike_included = (filterdf['VEHICLE TYPE CODE 1'].iloc[i] == 'Bike' or filterdf['VEHICLE TYPE CODE 2'].iloc[i] == 'Bike' or filterdf['VEHICLE TYPE CODE 3'].iloc[i] == 'Bike' or filterdf['VEHICLE TYPE CODE 4'].iloc[i] == 'Bike' or filterdf['VEHICLE TYPE CODE 5'].iloc[i] == 'Bike')
    Pedestrian_bike_error = (filterdf['CONTRIBUTING FACTOR VEHICLE 1'].iloc[i] == textt or filterdf['CONTRIBUTING FACTOR VEHICLE 2'].iloc[i] == textt or filterdf['CONTRIBUTING FACTOR VEHICLE 3'].iloc[i] == textt or filterdf['CONTRIBUTING FACTOR VEHICLE 4'].iloc[i] == textt or filterdf['CONTRIBUTING FACTOR VEHICLE 5'].iloc[i] == textt)
    if (filterdf['NUMBER OF CYCLIST INJURED'].iloc[i] + filterdf['NUMBER OF CYCLIST KILLED'].iloc[i]>0) or Bike_included:
        index_cyclist.append(i)
    elif (filterdf['NUMBER OF PEDESTRIANS INJURED'].iloc[i] + filterdf['NUMBER OF PEDESTRIANS KILLED'].iloc[i]>0 or (Pedestrian_bike_error and not(Bike_included))):
        index_pedestrian.append(i)    
df_pedestrian = filterdf.iloc[index_pedestrian,:]
df_cyclist = filterdf.iloc[index_cyclist,:]
df_cyclist.to_csv('df_cyclist.csv', index=False)
df_pedestrian.to_csv('df_pedestrian.csv', index=False)
# ## Start of notebook
# 

# In[66]:


df = pd.read_csv('data_preparred.csv',low_memory=False)
df['datetime']=pd.to_datetime(df['CRASH DATE'] + ' ' + df['CRASH TIME'])
df.datetime = pd.to_datetime(df.datetime, format='%Y-%m-%d %H:%M:%S')
df['month'] = df.datetime.apply(lambda x: x.month)
df['week'] = df.datetime.apply(lambda x: x.week)
df['day'] = df.datetime.apply(lambda x: x.day)
df['hour'] = df.datetime.apply(lambda x: x.hour)
df['year']= df.datetime.apply(lambda x: x.year)
df['day_of_week'] = df['datetime'].dt.dayofweek


# In[67]:


import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


# In[68]:


df.head()


# ## Time analysis of dataframe

# ### Year analysis

# In[69]:


date_wise=df.groupby(['year'])['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED'].agg('sum').reset_index()

date_wise['Total Accidents'] = df.groupby(['year']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
date_wise['Injury%'] = round((date_wise['NUMBER OF PERSONS INJURED']/date_wise['Total Accidents'] * 100), 1)
date_wise['Fatality%'] = round((date_wise['NUMBER OF PERSONS KILLED']/date_wise['Total Accidents'] * 100), 3)

date_wise = date_wise.sort_values('Total Accidents', ascending = False)


# In[70]:


date_wise = date_wise.sort_values('year', ascending = True)


# In[71]:


fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 10)))

# Creating the plot
date_wise.plot(x = 'year', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Total Accidents', size = 12)
ax.set_xlabel('Year', size = 12)
ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 10)

# Exporting the visualisation
fig.savefig('TotalAcci1221.png', dpi=500)


# In[72]:


# Create figure and axes for Matplotlib
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 10)))
color2 = cm.autumn(np.linspace(.2,.6, 10))

# Creating the plots
date_wise.plot(x = 'year', 
                      y = 'Injury%', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

date_wise.plot(x = 'year', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

# Customizing the Visulation
ax1.set_title('Injury % over the recent years', size = 12)
ax1.set_xlabel('Year', size = 12)
ax1.set_ylabel('Rate of Injury (%)', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Fatality over the recent years', size = 12)
ax2.set_xlabel('Year', size = 12)
ax2.set_ylabel('Rate of Fatality (%)', size = 12)
ax2.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('injfat1221.png', dpi=500)


# ### Month analysis

# In[73]:


date_wise=df.groupby(['month'])['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED'].agg('sum').reset_index()

date_wise['Total Accidents'] = df.groupby(['month']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
date_wise['Injury%'] = round((date_wise['NUMBER OF PERSONS INJURED']/date_wise['Total Accidents'] * 100), 1)
date_wise['Fatality%'] = round((date_wise['NUMBER OF PERSONS KILLED']/date_wise['Total Accidents'] * 100), 3)

date_wise = date_wise.sort_values('Total Accidents', ascending = False)


# In[74]:


date_wise = date_wise.sort_values('month', ascending = True)


# In[75]:


date_wise['month'].astype(str)


# In[76]:


date_wise["month"].replace({1: 'Jan', 2: 'Feb', 3: 'Mar',4: 'Apr', 5: 'May',6: 'Jun', 7:'Jul', 8: 'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}, inplace=True)


# In[77]:


fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 12)))

# Creating the plot
date_wise.plot(x = 'month', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Total Accidents', size = 12)
ax.set_xlabel('month', size = 12)

ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 10)

# Exporting the visualisation
fig.savefig('TotalAccimonthly.png', dpi=500)


# In[78]:


# Create figure and axes for Matplotlib
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 12)))
color2 = cm.autumn(np.linspace(.2,.6, 12))

# Creating the plots
date_wise.plot(x = 'month', 
                      y = 'Injury%', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

date_wise.plot(x = 'month', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

# Customizing the Visulation
ax1.set_title('Injury % over months', size = 12)
ax1.set_xlabel('Month', size = 12)
ax1.set_ylabel('Rate of Injury (%)', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Fatality over months', size = 12)
ax2.set_xlabel('Month', size = 12)
ax2.set_ylabel('Rate of Fatality (%)', size = 12)
ax2.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('injfatmonth.png', dpi=500)


# ### Weekday analysis

# In[79]:


date_wise=df.groupby(['day_of_week'])['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED'].agg('sum').reset_index()

date_wise['Total Accidents'] = df.groupby(['day_of_week']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
date_wise['Injury%'] = round((date_wise['NUMBER OF PERSONS INJURED']/date_wise['Total Accidents'] * 100), 1)
date_wise['Fatality%'] = round((date_wise['NUMBER OF PERSONS KILLED']/date_wise['Total Accidents'] * 100), 3)

date_wise = date_wise.sort_values('Total Accidents', ascending = False)


# In[80]:


date_wise = date_wise.sort_values('day_of_week', ascending = True)


# In[81]:


date_wise


# In[82]:


date_wise["day_of_week"].replace({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6:'Sunday'}, inplace=True)


# In[83]:


fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 12)))

# Creating the plot
date_wise.plot(x = 'day_of_week', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Total Accidents', size = 12)
ax.set_xlabel('Day of the week', size = 12)

ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 10)

# Exporting the visualisation
fig.savefig('TotalAccidaily.png', dpi=500)


# In[84]:


# Create figure and axes for Matplotlib
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 7)))
color2 = cm.autumn(np.linspace(.2,.6, 7))

# Creating the plots
date_wise.plot(x = 'day_of_week', 
                      y = 'Injury%', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

date_wise.plot(x = 'day_of_week', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

# Customizing the Visulation
ax1.set_title('Injury % over days of the week', size = 12)
ax1.set_xlabel('Day of the week', size = 12)
ax1.set_ylabel('Rate of Injury (%)', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Fatality over days of the week', size = 12)
ax2.set_xlabel('Day of the week', size = 12)
ax2.set_ylabel('Rate of Fatality (%)', size = 12)
ax2.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('injfatday.png', dpi=500)


# ### Accidents 24 hour span

# In[85]:


date_wise=df.groupby(['hour'])['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED'].agg('sum').reset_index()

date_wise['Total Accidents'] = df.groupby(['hour']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
date_wise['Injury%'] = round((date_wise['NUMBER OF PERSONS INJURED']/date_wise['Total Accidents'] * 100), 1)
date_wise['Fatality%'] = round((date_wise['NUMBER OF PERSONS KILLED']/date_wise['Total Accidents'] * 100), 3)

date_wise = date_wise.sort_values('Total Accidents', ascending = False)


# In[86]:


date_wise = date_wise.sort_values('hour', ascending = True)


# In[87]:


date_wise


# In[88]:


fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 24)))

# Creating the plot
date_wise.plot(x = 'hour', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Total Accidents', size = 12)
ax.set_xlabel('Hourly', size = 12)

ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 10)

# Exporting the visualisation
fig.savefig('TotalAccihourly.png', dpi=500)


# In[89]:


# Create figure and axes for Matplotlib
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 24)))
color2 = cm.autumn(np.linspace(.2,.6, 24))

# Creating the plots
date_wise.plot(x = 'hour', 
                      y = 'Injury%', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

date_wise.plot(x = 'hour', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

# Customizing the Visulation
ax1.set_title('Injury % over a 24 hour span', size = 12)
ax1.set_xlabel('Hourly', size = 12)
ax1.set_ylabel('Rate of Injury (%)', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Fatality over a 24 hour span ', size = 12)
ax2.set_xlabel('Hourly', size = 12)
ax2.set_ylabel('Rate of Fatality (%)', size = 12)
ax2.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('injfathourly.png', dpi=500)


# ## Type analysis

# In[90]:


# Calculate the number of people killed, injured and total accidents for each vehicle type
vehicle_wise = df.groupby(['VEHICLE TYPE CODE 1'])['NUMBER OF PERSONS KILLED', 'NUMBER OF PERSONS INJURED'].agg('sum').reset_index()

vehicle_wise['Total Accidents'] = df.groupby(['VEHICLE TYPE CODE 1']).size().reset_index(name='NoOfAccidents').NoOfAccidents

vehicle_wise = vehicle_wise.sort_values('Total Accidents', ascending = False)
# Injuries and Fatalities as Percentages
vehicle_wise['Injury%'] = round((vehicle_wise['NUMBER OF PERSONS INJURED']/vehicle_wise['Total Accidents'] * 100), 1)
vehicle_wise['Fatality%'] = round((vehicle_wise['NUMBER OF PERSONS KILLED']/vehicle_wise['Total Accidents'] * 100), 3)

# Filtering vehicles involved in atleast 100 accidents
mask = vehicle_wise['Total Accidents'] > 100
vehicle_wise = vehicle_wise[mask]

vehicle_wise.head(3)


# In[91]:


vehicle_accidents = vehicle_wise.sort_values('Total Accidents', ascending = False).head(10)
vehicle_accidents.head(3)


# In[92]:


from matplotlib import cm


# In[93]:


fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 10)))

# Creating the plot
vehicle_accidents.plot(x = 'VEHICLE TYPE CODE 1', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Vehicle types involved in the most number of accidents', size = 12)
ax.set_xlabel('Vehicle Type', size = 12)
ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 10)

# Exporting the visualisation
fig.savefig('vehicle_type_accidents.png', dpi=500)


# In[94]:


vehicle_inj_rate = vehicle_wise.sort_values('Injury%', ascending = False).head(10)
vehicle_fat_rate = vehicle_wise.sort_values('Fatality%', ascending = False).head(10)


# In[95]:


vehicle_fat_rate


# In[96]:


# Create figure and axes for Matplotlib
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 10)))
color2 = cm.autumn(np.linspace(.2,.6, 10))

# Creating the plots
vehicle_inj_rate.plot(x = 'VEHICLE TYPE CODE 1', 
                      y = 'Injury%', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

vehicle_fat_rate.plot(x = 'VEHICLE TYPE CODE 1', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

# Customizing the Visulation
ax1.set_title('Vehicle Types with the highest rate of injury', size = 12)
ax1.set_xlabel('Vehicle Types', size = 12)
ax1.set_ylabel('Rate of Injury (%)', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Vehicle Types with the highest rate of fatality', size = 12)
ax2.set_xlabel('Vehicle Types', size = 12)
ax2.set_ylabel('Rate of Fatality (%)', size = 12)
ax2.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('vehicle_inj_fat_rate.png', dpi=500)


# In[97]:


fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 10)))

vehicle_inj_rate_df=vehicle_inj_rate.sort_values('Total Accidents', ascending = False).head(10)

# Creating the plot
vehicle_inj_rate_df.plot(x = 'VEHICLE TYPE CODE 1', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax)


# Customizing the Visulation
ax.set_title('Vehicle types involved in the most number of accidents', size = 12)
ax.set_xlabel('Vehicle Type', size = 12)
ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 10)

# Exporting the visualisation
fig.savefig('vehicle_type_accidents.png', dpi=500)


# In[98]:


#https://www.kaggle.com/skhiearth/nyc-motor-vehicle-collisions


# Analysis: Passenger Vehicles are the vehicle types involved in the most number of accident, followed by Sedands and SUVs/Station Wagons. Bicycle and Motorcycle have an extreme rate of injury (>50%). Hence, it can be concluded that two wheelers are prone to injuries.

# ## Cyclist and Pedestrians analysis

# In[99]:


import matplotlib.pyplot as plt
import numpy as np


# In[ ]:





# In[100]:


df_cyclist = pd.read_csv('df_cyclist.csv',low_memory=False)
df_pedestrian = pd.read_csv('df_pedestrian.csv',low_memory=False)


# In[101]:


df_pedestrian.datetime = pd.to_datetime(df_pedestrian.datetime, format='%Y-%m-%d %H:%M:%S')
df_cyclist.datetime = pd.to_datetime(df_cyclist.datetime, format='%Y-%m-%d %H:%M:%S')


# In[102]:


df_pedestrian['hour'] = df_pedestrian.datetime.apply(lambda x: x.hour)
df_pedestrian['year']= df_pedestrian.datetime.apply(lambda x: x.year)
df_cyclist['hour'] = df_cyclist.datetime.apply(lambda x: x.hour)
df_cyclist['year']= df_cyclist.datetime.apply(lambda x: x.year)


# In[190]:


df_pedestrian['day_of_week'] = df_pedestrian['datetime'].dt.dayofweek
df_cyclist['day_of_week'] = df_pedestrian['datetime'].dt.dayofweek


# In[191]:


date_wisepd=df_pedestrian.groupby(['year'])['NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED'].agg('sum').reset_index()

date_wisepd['Total Accidents'] = df_pedestrian.groupby(['year']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
date_wisepd['Injury%'] = round((date_wisepd['NUMBER OF PEDESTRIANS INJURED']/date_wisepd['Total Accidents'] * 100), 1)
date_wisepd['Fatality%'] = round((date_wisepd['NUMBER OF PEDESTRIANS KILLED']/date_wisepd['Total Accidents'] * 100), 3)

date_wisepd = date_wisepd.sort_values('Total Accidents', ascending = False)


# In[192]:


date_wisepd = date_wisepd.sort_values('year', ascending = True)


# In[ ]:





# In[193]:


date_wisecy=df_cyclist.groupby(['year'])['NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED'].agg('sum').reset_index()

date_wisecy['Total Accidents'] = df_cyclist.groupby(['year']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
date_wisecy['Injury%'] = round((date_wisecy['NUMBER OF CYCLIST INJURED']/date_wisecy['Total Accidents'] * 100), 1)
date_wisecy['Fatality%'] = round((date_wisecy['NUMBER OF CYCLIST KILLED']/date_wisecy['Total Accidents'] * 100), 3)

date_wisecy = date_wisecy.sort_values('Total Accidents', ascending = False)


# In[194]:


date_wisecy = date_wisecy.sort_values('year', ascending = True)


# In[195]:


# Create figure and axes for Matplotlib
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 7)))
color2 = cm.autumn(np.linspace(.2,.6, 7))

# Creating the plots
date_wisepd.plot(x = 'year', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

date_wisecy.plot(x = 'year', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

# Customizing the Visulation
ax1.set_title('Total Accidents Pedestrians', size = 12)
ax1.set_xlabel('Year', size = 12)
ax1.set_ylabel('Number of Accidents', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Total Accidents Cyclist', size = 12)
ax2.set_xlabel('Year', size = 12)
ax2.set_ylabel('Number of Accidents', size = 12)
ax2.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('pdcyyear.png', dpi=500)


# ### Cyclist and Pedestrians analysis 24 hours

# In[196]:


date_wisepd=df_pedestrian.groupby(['hour'])['NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED'].agg('sum').reset_index()

date_wisepd['Total Accidents'] = df_pedestrian.groupby(['hour']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
date_wisepd['Injury%'] = round((date_wisepd['NUMBER OF PEDESTRIANS INJURED']/date_wisepd['Total Accidents'] * 100), 1)
date_wisepd['Fatality%'] = round((date_wisepd['NUMBER OF PEDESTRIANS KILLED']/date_wisepd['Total Accidents'] * 100), 3)

date_wisepd = date_wisepd.sort_values('Total Accidents', ascending = False)


# In[200]:


date_wisepd = date_wisepd.sort_values('hour', ascending = True)


# In[201]:


date_wisecy=df_cyclist.groupby(['hour'])['NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED'].agg('sum').reset_index()

date_wisecy['Total Accidents'] = df_cyclist.groupby(['hour']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
date_wisecy['Injury%'] = round((date_wisecy['NUMBER OF CYCLIST INJURED']/date_wisecy['Total Accidents'] * 100), 1)
date_wisecy['Fatality%'] = round((date_wisecy['NUMBER OF CYCLIST KILLED']/date_wisecy['Total Accidents'] * 100), 3)

date_wisecy = date_wisecy.sort_values('Total Accidents', ascending = False)


# In[202]:


date_wisecy = date_wisecy.sort_values('hour', ascending = True)


# In[ ]:





# In[ ]:





# In[203]:


# Create figure and axes for Matplotlib
#fig, ax = plt.subplots(1, figsize=(14, 6))

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
fig1, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 24)))
color2 = cm.autumn(np.linspace(.2,.6, 24))
color3 = np.flip(cm.plasma(np.linspace(.2,.6, 24)))
color4 = cm.autumn(np.linspace(.2,.6, 24))

# Creating the plots
date_wisepd.plot(x = 'hour', 
                      y = 'Injury%', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

date_wisepd.plot(x = 'hour', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)
date_wisecy.plot(x = 'hour', 
                      y = 'Injury%', kind = 'bar', 
                      color = color3, stacked = True, ax = ax3)

date_wisecy.plot(x = 'hour', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color4, stacked = True, ax = ax4)

# Customizing the Visulation
ax1.set_title('Injury % over a 24 hour span Pedestrian ', size = 12)
ax1.set_xlabel('Hourly', size = 12)
ax1.set_ylabel('Rate of Injury (%)', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Fatality over a 24 hour span Pedestrian  ', size = 12)
ax2.set_xlabel('Hourly', size = 12)
ax2.set_ylabel('Rate of Fatality (%)', size = 12)
ax2.tick_params(labelrotation = 30)

ax3.set_title('Injury % over a 24 hour span Cyclist', size = 12)
ax3.set_xlabel('Hourly', size = 12)
ax3.set_ylabel('Rate of Injury (%)', size = 12)
ax3.tick_params(labelrotation = 30)

ax4.set_title('Fatality over a 24 hour span Cyclist ', size = 12)
ax4.set_xlabel('Hourly', size = 12)
ax4.set_ylabel('Rate of Fatality (%)', size = 12)
ax4.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('injfathourlypdcy.png', dpi=500)
fig1.savefig('injfathourlypdcy1.png', dpi=500)


# In[ ]:





# In[ ]:





# In[217]:


date_wise=df_pedestrian.groupby(['day_of_week'])['NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED'].agg('sum').reset_index()

date_wise['Total Accidents'] = df_pedestrian.groupby(['day_of_week']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
date_wise['Injury%'] = round((date_wise['NUMBER OF PEDESTRIANS INJURED']/date_wise['Total Accidents'] * 100), 1)
date_wise['Fatality%'] = round((date_wise['NUMBER OF PEDESTRIANS KILLED']/date_wise['Total Accidents'] * 100), 3)

date_wise = date_wise.sort_values('Total Accidents', ascending = False)


# In[218]:


date_wise = date_wise.sort_values('day_of_week', ascending = True)


# In[219]:


date_wise


# In[220]:


date_wise["day_of_week"].replace({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6:'Sunday'}, inplace=True)


# In[221]:


fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 12)))

# Creating the plot
date_wise.plot(x = 'day_of_week', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Total Accidents', size = 12)
ax.set_xlabel('Day of the week', size = 12)

ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 10)

# Exporting the visualisation
fig.savefig('TotalAccidaily.png', dpi=500)


# In[222]:


# Create figure and axes for Matplotlib
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 7)))
color2 = cm.autumn(np.linspace(.2,.6, 7))

# Creating the plots
date_wise.plot(x = 'day_of_week', 
                      y = 'Injury%', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

date_wise.plot(x = 'day_of_week', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

# Customizing the Visulation
ax1.set_title('Injury % over days of the week', size = 12)
ax1.set_xlabel('Day of the week', size = 12)
ax1.set_ylabel('Rate of Injury (%)', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Fatality over days of the week', size = 12)
ax2.set_xlabel('Day of the week', size = 12)
ax2.set_ylabel('Rate of Fatality (%)', size = 12)
ax2.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('injfatdaypd.png', dpi=500)


# In[223]:


date_wise=df_cyclist.groupby(['day_of_week'])['NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED'].agg('sum').reset_index()

date_wise['Total Accidents'] = df_pedestrian.groupby(['day_of_week']).size().reset_index(name='NoOfAccidents').NoOfAccidents

# Injuries and Fatalities as Percentages
date_wise['Injury%'] = round((date_wise['NUMBER OF CYCLIST INJURED']/date_wise['Total Accidents'] * 100), 1)
date_wise['Fatality%'] = round((date_wise['NUMBER OF CYCLIST KILLED']/date_wise['Total Accidents'] * 100), 3)

date_wise = date_wise.sort_values('Total Accidents', ascending = False)


# In[224]:


date_wise = date_wise.sort_values('day_of_week', ascending = True)


# In[225]:


date_wise


# In[226]:


date_wise["day_of_week"].replace({0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6:'Sunday'}, inplace=True)


# In[227]:


fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 12)))

# Creating the plot
date_wise.plot(x = 'day_of_week', 
                      y = 'Total Accidents', kind = 'bar', 
                      color = color, stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Total Accidents', size = 12)
ax.set_xlabel('Day of the week', size = 12)

ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 10)

# Exporting the visualisation
fig.savefig('TotalAccidaily.png', dpi=500)


# In[228]:


# Create figure and axes for Matplotlib
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 7)))
color2 = cm.autumn(np.linspace(.2,.6, 7))

# Creating the plots
date_wise.plot(x = 'day_of_week', 
                      y = 'Injury%', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

date_wise.plot(x = 'day_of_week', 
                      y = 'Fatality%', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

# Customizing the Visulation
ax1.set_title('Injury % over days of the week', size = 12)
ax1.set_xlabel('Day of the week', size = 12)
ax1.set_ylabel('Rate of Injury (%)', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Fatality over days of the week', size = 12)
ax2.set_xlabel('Day of the week', size = 12)
ax2.set_ylabel('Rate of Fatality (%)', size = 12)
ax2.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('injfatdaycy.png', dpi=500)


# In[ ]:





# ## Location analysis

# In[113]:


dfmap=df.groupby(['ZIP CODE'], as_index=False).sum()


# In[114]:


#dfmap


# In[115]:


# Import the ShapeFile for Borough Boundaries
fp = 'ZIP_CODE_040114.shp'
zip_geo = gpd.read_file(fp)
#borough_geo['boro_name'] = borough_geo['boro_name'].str.upper() 
#https://jsspina.carto.com/tables/nyc_zip_code_tabulation_areas_polygons/public/map
# Merging ShapeFile with data
#borough_wise = borough_geo.set_index('boro_name').join(df2.set_index('BOROUGH'))


# In[116]:


#zip_geo.head(5)


# In[117]:


zip_geo.rename(columns = {'ZIPCODE':'ZIP CODE'}, inplace = True)


# In[118]:


zip_geo.head(2)


# In[119]:


zip_geo=zip_geo[['ZIP CODE','PO_NAME','geometry','COUNTY']]


# In[ ]:





# In[120]:


dfcyclemerged=zip_geo.merge(dfmap,on="ZIP CODE")


# In[121]:


nymap = folium.Map(location=[40.72, -74.000], zoom_start=11,tiles=None)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(nymap)


# In[122]:


NIL=folium.Choropleth(
    geo_data=dfcyclemerged,
    name="injuries",
    data=dfcyclemerged,
    columns=["ZIP CODE", "NUMBER OF PERSONS INJURED"],
    key_on="feature.properties.ZIP CODE",
    fill_color="OrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="NUMBER OF PERSONS INJURED",
)

nymap.add_child(NIL)
nymap.keep_in_front(NIL)
folium.LayerControl().add_to(nymap)
nymap


# In[123]:






style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

NIL = folium.features.GeoJson(
    dfcyclemerged,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['PO_NAME','NUMBER OF PERSONS INJURED',"NUMBER OF PERSONS KILLED",'NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED'],
        aliases=['Neighborhood: ','Injuries Total','Deaths Total',"Pedestrains injuries","Pedestrains deaths", "Cyclist injuries","Cyclist Deaths"],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)
nymap.add_child(NIL)
nymap.keep_in_front(NIL)
folium.LayerControl().add_to(nymap)
nymap

outfp = "choropleth_map1.html"
nymap.save(outfp)


# In[124]:


nymap


# In[125]:


nymap = folium.Map(location=[40.72, -74.000], zoom_start=11,tiles=None)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(nymap)


# In[126]:


nymap = folium.Map(location=[40.72, -74.000], zoom_start=11,tiles=None)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(nymap)
NIL=folium.Choropleth(
    geo_data=dfcyclemerged,
    name="injuries",
    data=dfcyclemerged,
    columns=["ZIP CODE", "NUMBER OF PEDESTRIANS INJURED"],
    key_on="feature.properties.ZIP CODE",
    fill_color="OrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="NUMBER OF PEDESTRIANS INJURED",
)

nymap.add_child(NIL)
nymap.keep_in_front(NIL)
folium.LayerControl().add_to(nymap)
nymap


# In[127]:






style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

NIL = folium.features.GeoJson(
    dfcyclemerged,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['PO_NAME','NUMBER OF PERSONS INJURED',"NUMBER OF PERSONS KILLED",'NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED'],
        aliases=['Neighborhood: ','Injuries Total','Deaths Total',"Pedestrains injuries","Pedestrains deaths", "Cyclist injuries","Cyclist Deaths"],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)
nymap.add_child(NIL)
nymap.keep_in_front(NIL)
folium.LayerControl().add_to(nymap)
nymap

outfp = "choropleth_map2.html"
nymap.save(outfp)


# In[128]:


nymap


# In[129]:


nymap = folium.Map(location=[40.72, -74.000], zoom_start=11,tiles=None)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(nymap)


# In[130]:


nymap = folium.Map(location=[40.72, -74.000], zoom_start=11,tiles=None)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(nymap)
NIL=folium.Choropleth(
    geo_data=dfcyclemerged,
    name="injuries",
    data=dfcyclemerged,
    columns=["ZIP CODE", "NUMBER OF CYCLIST INJURED"],
    key_on="feature.properties.ZIP CODE",
    fill_color="OrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="NUMBER OF CYCLIST INJURED",
)

nymap.add_child(NIL)
nymap.keep_in_front(NIL)
folium.LayerControl().add_to(nymap)
nymap


# In[131]:






style_function = lambda x: {'fillColor': '#ffffff', 
                            'color':'#000000', 
                            'fillOpacity': 0.1, 
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000', 
                                'color':'#000000', 
                                'fillOpacity': 0.50, 
                                'weight': 0.1}

NIL = folium.features.GeoJson(
    dfcyclemerged,
    style_function=style_function, 
    control=False,
    highlight_function=highlight_function, 
    tooltip=folium.features.GeoJsonTooltip(
        fields=['PO_NAME','NUMBER OF PERSONS INJURED',"NUMBER OF PERSONS KILLED",'NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED','NUMBER OF CYCLIST KILLED','NUMBER OF CYCLIST KILLED'],
        aliases=['Neighborhood: ','Injuries Total','Deaths Total',"Pedestrains injuries","Pedestrains deaths", "Cyclist injuries","Cyclist Deaths"],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;") 
    )
)
nymap.add_child(NIL)
nymap.keep_in_front(NIL)
folium.LayerControl().add_to(nymap)
nymap

outfp = "choropleth_map3.html"
nymap.save(outfp)


# In[132]:


nymap


# In[ ]:





# In[ ]:





# ## Weather data analysis

# In[133]:


df['just_date'] = df['datetime'].dt.date


# In[134]:


weather = pd.read_csv('2577799.csv',low_memory=False)


# In[135]:


weather.head()


# In[136]:


weather.columns


# In[137]:


weatherdate=weather[['DATE','AWND', 'PGTM', 'PRCP', 'SNOW', 'SNWD',
       'TAVG', 'TMAX', 'TMIN', 'TSUN']]


# In[138]:


weatherdate


# In[139]:


weatherdate.DATE = pd.to_datetime(weatherdate.DATE, format='%Y-%m-%d %H:%M:%S')


# In[140]:


weatherdate['just_date'] = weatherdate['DATE'].dt.date


# In[141]:


weatherdate


# In[142]:


drydays=weatherdate.loc[(weatherdate['PRCP']==0) & (weatherdate['TMIN'] > 0)].shape[0]
drydaysm=weatherdate.loc[(weatherdate['PRCP']==0) & (weatherdate['TMIN'] <= 0)].shape[0]
rainligth=weatherdate.loc[(weatherdate['PRCP']>0) & (weatherdate['PRCP']<=10)].shape[0]
rainheavy=weatherdate.loc[(weatherdate['PRCP']>10)].shape[0]
snowheavy=weatherdate.loc[(weatherdate['SNOW']>10)].shape[0]
snowligth=weatherdate.loc[(weatherdate['SNOW']>0) & (weatherdate['SNOW']<=10)].shape[0]


# In[143]:


drydays,drydaysm,rainligth,rainheavy,snowheavy,snowligth


# In[144]:


df1=weatherdate.merge(df,on="just_date")


# In[145]:


weatherdate['SNOW'].describe()


# In[ ]:





# In[146]:


def weather(n,y,x):
    if n ==0 and y==0 and x>= 0.0:
        return 'Dry road+'
    elif n ==0 and y==0 and x< 0.0:
        return 'Dry road-'
    elif 0.0 < y <= 10:
        return 'Ligth snowy road'
    elif y > 10:
        return 'Heavy snowy road'
    elif 0.0 < n <= 10 :
        return 'Ligth wet road'
    elif n > 10 :
        return 'Heavy wet road'
   
    
    
df1['Weather type']=df1.apply(lambda x: weather(x['PRCP'], x['SNOW'],x['TMIN']), axis=1)


# In[147]:


df1['Weather type'].value_counts()


# In[ ]:





# In[ ]:





# In[148]:


to_plot_grouped = df1.groupby(['Weather type'])['NUMBER OF PERSONS KILLED', 'NUMBER OF PERSONS INJURED'].agg('sum').reset_index()


to_plot_grouped['Total Accidents'] = df1.groupby(['Weather type']).size().reset_index(name='NoOfAccidents').NoOfAccidents
to_plot_grouped['daysof']=drydays,drydaysm,snowheavy,rainheavy,snowligth,rainligth


# Injuries and Fatalities as Percentages
to_plot_grouped['Average injuries per day'] = to_plot_grouped['NUMBER OF PERSONS INJURED']/to_plot_grouped['daysof']
to_plot_grouped['Average fatalities per day'] = to_plot_grouped['NUMBER OF PERSONS KILLED']/to_plot_grouped['daysof']

to_plot_grouped = to_plot_grouped.sort_values('Total Accidents', ascending = False)
# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 10)))

# Creating the plot
to_plot_grouped.plot(x = 'Weather type', y = 'Total Accidents', 
             kind = 'bar', color = color, 
             stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Weather Condition vs Number of Accidents', size = 12)
ax.set_xlabel('Weather Condition', size = 12)
ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 90)

# Exporting the visualisation
fig.savefig('weather_summary_accidents.png', dpi=500)


# In[ ]:





# In[ ]:





# In[149]:


weather_inj_rate = to_plot_grouped.sort_values('Average injuries per day', ascending = False).head(10)
weather_inj = to_plot_grouped.sort_values('NUMBER OF PERSONS INJURED', ascending = False).head(10)
weather_fat_rate = to_plot_grouped.sort_values('Average fatalities per day', ascending = False).head(10)
weather_fat = to_plot_grouped.sort_values('NUMBER OF PERSONS KILLED', ascending = False).head(10)
# Create figure and axes for Matplotlib
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))
fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(18, 18))
fig.tight_layout(pad=10.0)

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 10)))
color2 = np.flip(cm.plasma(np.linspace(.2,.6, 10)))

# Creating the plots
weather_inj_rate.plot(x = 'Weather type', 
                      y = 'Average injuries per day', kind = 'bar', 
                      color = color, stacked = True, ax = ax2)

#weather_inj.plot(x = 'Weather type', y = 'NUMBER OF PERSONS INJURED', kind = 'bar', 
                # color = color, stacked = True, ax = ax3)

weather_fat_rate.plot(x = 'Weather type', 
                      y = 'Average fatalities per day', kind = 'bar', 
                      color = color2, stacked = True, ax = ax3)

weather_fat.plot(x = 'Weather type', y = 'daysof', kind = 'bar', 
                 color = color2, stacked = True, ax = ax1)

# Customizing the Visulation

ax1.set_title('Number of days with Weather Condition', size = 12)
ax1.set_xlabel('Weather Condition', size = 12)
ax1.set_ylabel('Number of Days', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Weather Condition with the highest rate of injury', size = 12)
ax2.set_xlabel(' ', size = 12)
ax2.set_ylabel('Averge injuryes per day', size = 12)
ax2.tick_params(labelrotation = 30)

ax3.set_title('Weather Condition with the highest rate of fatality', size = 12)
ax3.set_xlabel(' ', size = 12)
ax3.set_ylabel('Average of fatality per day', size = 12)
ax3.tick_params(labelrotation = 30)



#ax4.set_title('Weather Condition vs Fatalities', size = 12)
#ax4.set_xlabel('Weather Condition', size = 12)
#ax4.set_ylabel('Number of Deaths', size = 12)
#ax4.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('weather_inj_fat_rate.png', dpi=500)


# ### Weather data vs cyclists accidents

# In[150]:


df_cyclist['just_date'] = df_cyclist['datetime'].dt.date


# In[151]:


df2=weatherdate.merge(df_cyclist,on="just_date")


# In[152]:


def weather(n,y,x):
    if n ==0 and y==0 and x>= 0.0:
        return 'Dry road+'
    elif n ==0 and y==0 and x< 0.0:
        return 'Dry road-'
    elif 0.0 < y <= 10:
        return 'Ligth snowy road'
    elif y > 10:
        return 'Heavy snowy road'
    elif 0.0 < n <= 10 :
        return 'Ligth wet road'
    elif n > 10 :
        return 'Heavy wet road'
   
    
    
df2['Weather type']=df2.apply(lambda x: weather(x['PRCP'], x['SNOW'],x['TMIN']), axis=1)


# In[153]:


df2


# In[154]:


to_plot_grouped = df2.groupby(['Weather type'])['NUMBER OF PERSONS KILLED', 'NUMBER OF PERSONS INJURED'].agg('sum').reset_index()


to_plot_grouped['Total Accidents'] = df2.groupby(['Weather type']).size().reset_index(name='NoOfAccidents').NoOfAccidents
to_plot_grouped['daysof']=drydays,drydaysm,snowheavy,rainheavy,snowligth,rainligth


# Injuries and Fatalities as Percentages
to_plot_grouped['Average injuries per day for cyclist'] = to_plot_grouped['NUMBER OF PERSONS INJURED']/to_plot_grouped['daysof']
to_plot_grouped['Average fatalities per day for cyclist'] = to_plot_grouped['NUMBER OF PERSONS KILLED']/to_plot_grouped['daysof']

to_plot_grouped = to_plot_grouped.sort_values('Total Accidents', ascending = False)
# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 10)))

# Creating the plot
to_plot_grouped.plot(x = 'Weather type', y = 'Total Accidents', 
             kind = 'bar', color = color, 
             stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Weather Condition vs Number of Accidents', size = 12)
ax.set_xlabel('Weather Condition', size = 12)
ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 90)

# Exporting the visualisation
fig.savefig('weather_summary_accidentscyc.png', dpi=500)


# In[ ]:





# In[155]:


weather_inj_rate = to_plot_grouped.sort_values('Average injuries per day for cyclist', ascending = False).head(10)
weather_inj = to_plot_grouped.sort_values('NUMBER OF PERSONS INJURED', ascending = False).head(10)
weather_fat_rate = to_plot_grouped.sort_values('Average fatalities per day for cyclist', ascending = False).head(10)
weather_fat = to_plot_grouped.sort_values('NUMBER OF PERSONS KILLED', ascending = False).head(10)
# Create figure and axes for Matplotlib
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 10)))
color2 = np.flip(cm.plasma(np.linspace(.2,.6, 10)))

# Creating the plots
weather_inj_rate.plot(x = 'Weather type', 
                      y = 'Average injuries per day for cyclist', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

weather_inj.plot(x = 'Weather type', y = 'NUMBER OF PERSONS INJURED', kind = 'bar', 
                 color = color, stacked = True, ax = ax3)

weather_fat_rate.plot(x = 'Weather type', 
                      y = 'Average fatalities per day for cyclist', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

weather_fat.plot(x = 'Weather type', y = 'NUMBER OF PERSONS KILLED', kind = 'bar', 
                 color = color2, stacked = True, ax = ax4)

# Customizing the Visulation
ax1.set_title('Weather Condition with the highest rate of injury', size = 12)
ax1.set_xlabel(' ', size = 12)
ax1.set_ylabel('Averge injuryes per day', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Weather Condition with the highest rate of fatality', size = 12)
ax2.set_xlabel(' ', size = 12)
ax2.set_ylabel('Average of fatality per day', size = 12)
ax2.tick_params(labelrotation = 30)

ax3.set_title('Weather Condition vs Injuries', size = 12)
ax3.set_xlabel('Weather Condition', size = 12)
ax3.set_ylabel('Number of Injured People', size = 12)
ax3.tick_params(labelrotation = 30)

ax4.set_title('Weather Condition vs Fatalities', size = 12)
ax4.set_xlabel('Weather Condition', size = 12)
ax4.set_ylabel('Number of Deaths', size = 12)
ax4.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('weather_inj_fat_ratecyc.png', dpi=500)


# ### Weather data vs Pedestrians accidents

# In[156]:


df_pedestrian['just_date'] = df_pedestrian['datetime'].dt.date

df3=weatherdate.merge(df_pedestrian,on="just_date")

def weather(n,y,x):
    if n ==0 and y==0 and x>= 0.0:
        return 'Dry road+'
    elif n ==0 and y==0 and x< 0.0:
        return 'Dry road-'
    elif 0.0 < y <= 10:
        return 'Ligth snowy road'
    elif y > 10:
        return 'Heavy snowy road'
    elif 0.0 < n <= 10 :
        return 'Ligth wet road'
    elif n > 10 :
        return 'Heavy wet road'
   
    
    
df3['Weather type']=df3.apply(lambda x: weather(x['PRCP'], x['SNOW'],x['TMIN']), axis=1)


to_plot_grouped = df3.groupby(['Weather type'])['NUMBER OF PERSONS KILLED', 'NUMBER OF PERSONS INJURED'].agg('sum').reset_index()


to_plot_grouped['Total Accidents'] = df3.groupby(['Weather type']).size().reset_index(name='NoOfAccidents').NoOfAccidents
to_plot_grouped['daysof']=drydays,drydaysm,snowheavy,rainheavy,snowligth,rainligth


# Injuries and Fatalities as Percentages
to_plot_grouped['Average injuries per day for pedestrians'] = to_plot_grouped['NUMBER OF PERSONS INJURED']/to_plot_grouped['daysof']
to_plot_grouped['Average fatalities per day for pedestrians'] = to_plot_grouped['NUMBER OF PERSONS KILLED']/to_plot_grouped['daysof']

to_plot_grouped = to_plot_grouped.sort_values('Total Accidents', ascending = False)
# Create figure and axes for Matplotlib
fig, ax = plt.subplots(1, figsize=(14, 6))

# Defining color map
color = np.flip(cm.Reds(np.linspace(.2,.6, 10)))

# Creating the plot
to_plot_grouped.plot(x = 'Weather type', y = 'Total Accidents', 
             kind = 'bar', color = color, 
             stacked = True, ax = ax)

# Customizing the Visulation
ax.set_title('Weather Condition vs Number of Accidents', size = 12)
ax.set_xlabel('Weather Condition', size = 12)
ax.set_ylabel('Number of Accidents', size = 12)
ax.tick_params(labelrotation = 90)

# Exporting the visualisation
fig.savefig('weather_summary_accidentspd.png', dpi=500)




# In[ ]:





# In[157]:




weather_inj_rate = to_plot_grouped.sort_values('Average injuries per day for pedestrians', ascending = False).head(10)
weather_inj = to_plot_grouped.sort_values('NUMBER OF PERSONS INJURED', ascending = False).head(10)
weather_fat_rate = to_plot_grouped.sort_values('Average fatalities per day for pedestrians', ascending = False).head(10)
weather_fat = to_plot_grouped.sort_values('NUMBER OF PERSONS KILLED', ascending = False).head(10)
# Create figure and axes for Matplotlib
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(18, 18))

# Defining color map
color = np.flip(cm.plasma(np.linspace(.2,.6, 10)))
color2 = np.flip(cm.plasma(np.linspace(.2,.6, 10)))

# Creating the plots
weather_inj_rate.plot(x = 'Weather type', 
                      y = 'Average injuries per day for pedestrians', kind = 'bar', 
                      color = color, stacked = True, ax = ax1)

weather_inj.plot(x = 'Weather type', y = 'NUMBER OF PERSONS INJURED', kind = 'bar', 
                 color = color, stacked = True, ax = ax3)

weather_fat_rate.plot(x = 'Weather type', 
                      y = 'Average fatalities per day for pedestrians', kind = 'bar', 
                      color = color2, stacked = True, ax = ax2)

weather_fat.plot(x = 'Weather type', y = 'NUMBER OF PERSONS KILLED', kind = 'bar', 
                 color = color2, stacked = True, ax = ax4)

# Customizing the Visulation
ax1.set_title('Weather Condition with the highest rate of injury', size = 12)
ax1.set_xlabel(' ', size = 12)
ax1.set_ylabel('Averge injuryes per day', size = 12)
ax1.tick_params(labelrotation = 30)

ax2.set_title('Weather Condition with the highest rate of fatality', size = 12)
ax2.set_xlabel(' ', size = 12)
ax2.set_ylabel('Average of fatality per day', size = 12)
ax2.tick_params(labelrotation = 30)

ax3.set_title('Weather Condition vs Injuries', size = 12)
ax3.set_xlabel('Weather Condition', size = 12)
ax3.set_ylabel('Number of Injured People', size = 12)
ax3.tick_params(labelrotation = 30)

ax4.set_title('Weather Condition vs Fatalities', size = 12)
ax4.set_xlabel('Weather Condition', size = 12)
ax4.set_ylabel('Number of Deaths', size = 12)
ax4.tick_params(labelrotation = 30)

# Exporting the visualisation
fig.savefig('weather_inj_fat_rateped.png', dpi=500)


# ## Location & Time 

# In[158]:


from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import plotly.graph_objs as go


import plotly.offline as offline

from plotly.graph_objs import *

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


# In[ ]:





# In[ ]:





# In[159]:


dfcyclemergedtime=df


# In[160]:


dfcyclegrouptime=dfcyclemergedtime.groupby(['ZIP CODE','hour'], as_index=False).sum()


# In[161]:


dfcyclegrouptime=zip_geo.merge(dfcyclegrouptime,on="ZIP CODE")


# ### General

# In[162]:


dfcyclemergedtime=df


# In[ ]:





# In[163]:


dfcyclegrouptime=dfcyclemergedtime.groupby(['ZIP CODE','hour'], as_index=False).sum()


# In[164]:


dfcyclegrouptime=zip_geo.merge(dfcyclegrouptime,on="ZIP CODE")


# In[165]:


dfcyclegrouptime=dfcyclegrouptime[['ZIP CODE','hour','PO_NAME','geometry','COUNTY','NUMBER OF PERSONS INJURED','NUMBER OF PERSONS KILLED']]


# In[166]:


nymap1 = folium.Map(location=[40.72, -74.000], zoom_start=11,tiles=None)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(nymap1)
nymap1


# In[167]:


ALL = 'ALL'
def unique_sorted_values_plus_ALL(array):
    unique = array.unique().tolist()
    unique.sort()
    unique.insert(0, ALL)
    return unique
dropdown_year = widgets.SelectionSlider(options = unique_sorted_values_plus_ALL(dfcyclegrouptime.hour))
output_year=widgets.Output()


# In[168]:


dropdown_year = widgets.SelectionSlider(options = unique_sorted_values_plus_ALL(dfcyclegrouptime.hour))
output_year=widgets.Output()


# In[169]:


#dfcyclegrouptime.to_csv('dfcyclegrouptime.csv', index=False)


# In[170]:


def dropdown_year_eventhandler(change):
    output_year.clear_output()
    with output_year:
        nymap1 = folium.Map(location=[40.72, -74.000], zoom_start=11,tiles=None)
        folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(nymap1)
        if (change.new == ALL):
            NIL=folium.Choropleth(
                geo_data=dfcyclegrouptime,
                name="injuries",
                data=dfcyclegrouptime,
                columns=["ZIP CODE", "NUMBER OF PERSONS KILLED"],
                key_on="feature.properties.ZIP CODE",
                fill_color="OrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="NUMBER OF PERSONS KILLED",)

            nymap1.add_child(NIL)
            nymap1.keep_in_front(NIL)
            folium.LayerControl().add_to(nymap1)
            #return nymap1
            display(nymap1)
        else:
            dfgeo=dfcyclegrouptime[dfcyclegrouptime.hour == change.new]
            NIL=folium.Choropleth(
                geo_data=dfgeo,
                name="injuries",
                data=dfgeo,
                columns=["ZIP CODE", "NUMBER OF PERSONS KILLED"],
                key_on="feature.properties.ZIP CODE",
                fill_color="OrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="NUMBER OF PERSONS KILLED",
            )
            nymap1.add_child(NIL)
            nymap1.keep_in_front(NIL)
            folium.LayerControl().add_to(nymap1)
            display(nymap1)
            #return nymap1
            #display(dfcyclegrouptime[dfcyclegrouptime.hour == change.new])
            #print(change.new)
                    
dropdown_year.observe(dropdown_year_eventhandler, names='value')
#display(nymap)
outfp = "choropleth_map4.html"
nymap1.save(outfp)
display(dropdown_year)              
display(output_year)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[171]:


display(output_year)


# In[ ]:





# ### Cyclist

# In[172]:


dfcyclemergedtime=df

dfcyclegrouptime=dfcyclemergedtime.groupby(['ZIP CODE','hour'], as_index=False).sum()

dfcyclegrouptime=zip_geo.merge(dfcyclegrouptime,on="ZIP CODE")


# In[173]:


dfcyclegrouptime=dfcyclegrouptime[['ZIP CODE','hour','PO_NAME','geometry','COUNTY','NUMBER OF CYCLIST INJURED','NUMBER OF CYCLIST KILLED']]


# In[174]:


nymap1 = folium.Map(location=[40.72, -74.000], zoom_start=11,tiles=None)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(nymap1)
nymap1


# In[175]:


ALL = 'ALL'
def unique_sorted_values_plus_ALL(array):
    unique = array.unique().tolist()
    unique.sort()
    unique.insert(0, ALL)
    return unique
dropdown_year = widgets.SelectionSlider(options = unique_sorted_values_plus_ALL(dfcyclegrouptime.hour))
output_year=widgets.Output()


# In[176]:


dropdown_year = widgets.SelectionSlider(options = unique_sorted_values_plus_ALL(dfcyclegrouptime.hour))
output_year=widgets.Output()


# In[177]:


def dropdown_year_eventhandler(change):
    output_year.clear_output()
    with output_year:
        nymap1 = folium.Map(location=[40.72, -74.000], zoom_start=11,tiles=None)
        folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(nymap1)
        if (change.new == ALL):
            NIL=folium.Choropleth(
                geo_data=dfcyclegrouptime,
                name="injuries",
                data=dfcyclegrouptime,
                columns=["ZIP CODE", "NUMBER OF CYCLIST KILLED"],
                key_on="feature.properties.ZIP CODE",
                fill_color="OrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="NUMBER OF CYCLIST KILLED",)

            nymap1.add_child(NIL)
            nymap1.keep_in_front(NIL)
            folium.LayerControl().add_to(nymap1)
            #return nymap1
            display(nymap1)
        else:
            dfgeo=dfcyclegrouptime[dfcyclegrouptime.hour == change.new]
            NIL=folium.Choropleth(
                geo_data=dfgeo,
                name="injuries",
                data=dfgeo,
                columns=["ZIP CODE", "NUMBER OF CYCLIST KILLED"],
                key_on="feature.properties.ZIP CODE",
                fill_color="OrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="NUMBER OF CYCLIST KILLED",
            )
            nymap1.add_child(NIL)
            nymap1.keep_in_front(NIL)
            folium.LayerControl().add_to(nymap1)
            display(nymap1)
            #return nymap1
            #display(dfcyclegrouptime[dfcyclegrouptime.hour == change.new])
            #print(change.new)
                    
dropdown_year.observe(dropdown_year_eventhandler, names='value')
#display(nymap)
outfp = "choropleth_map4.html"
nymap1.save(outfp)
display(dropdown_year)              


# In[ ]:





# In[ ]:





# In[178]:


display(output_year)


# ### Pedestrians 

# In[179]:


dfcyclemergedtime=df


# In[180]:


dfcyclegrouptime=dfcyclemergedtime.groupby(['ZIP CODE','hour'], as_index=False).sum()


# In[181]:


dfcyclegrouptime=zip_geo.merge(dfcyclegrouptime,on="ZIP CODE")


# In[ ]:





# In[182]:


dfcyclegrouptime=dfcyclegrouptime[['ZIP CODE','hour','PO_NAME','geometry','COUNTY','NUMBER OF PEDESTRIANS INJURED','NUMBER OF PEDESTRIANS KILLED']]


# In[183]:


nymap1 = folium.Map(location=[40.72, -74.000], zoom_start=11,tiles=None)
folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(nymap1)
nymap1


# In[184]:


ALL = 'ALL'
def unique_sorted_values_plus_ALL(array):
    unique = array.unique().tolist()
    unique.sort()
    unique.insert(0, ALL)
    return unique
dropdown_year = widgets.SelectionSlider(options = unique_sorted_values_plus_ALL(dfcyclegrouptime.hour))
output_year=widgets.Output()


# In[185]:


dropdown_year = widgets.SelectionSlider(options = unique_sorted_values_plus_ALL(dfcyclegrouptime.hour))
output_year=widgets.Output()


# In[186]:


#dfcyclegrouptime.to_csv('dfcyclegrouptime.csv', index=False)


# In[187]:


def dropdown_year_eventhandler(change):
    output_year.clear_output()
    with output_year:
        nymap1 = folium.Map(location=[40.72, -74.000], zoom_start=11,tiles=None)
        folium.TileLayer('CartoDB positron',name="Light Map",control=False).add_to(nymap1)
        if (change.new == ALL):
            NIL=folium.Choropleth(
                geo_data=dfcyclegrouptime,
                name="injuries",
                data=dfcyclegrouptime,
                columns=["ZIP CODE", "NUMBER OF PEDESTRIANS KILLED"],
                key_on="feature.properties.ZIP CODE",
                fill_color="OrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="NUMBER OF PEDESTRIANS KILLED",)

            nymap1.add_child(NIL)
            nymap1.keep_in_front(NIL)
            folium.LayerControl().add_to(nymap1)
            #return nymap1
            display(nymap1)
        else:
            dfgeo=dfcyclegrouptime[dfcyclegrouptime.hour == change.new]
            NIL=folium.Choropleth(
                geo_data=dfgeo,
                name="injuries",
                data=dfgeo,
                columns=["ZIP CODE", "NUMBER OF PEDESTRIANS KILLED"],
                key_on="feature.properties.ZIP CODE",
                fill_color="OrRd",
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name="NUMBER OF PEDESTRIANS KILLED",
            )
            nymap1.add_child(NIL)
            nymap1.keep_in_front(NIL)
            folium.LayerControl().add_to(nymap1)
            display(nymap1)
            #return nymap1
            #display(dfcyclegrouptime[dfcyclegrouptime.hour == change.new])
            #print(change.new)
                    
dropdown_year.observe(dropdown_year_eventhandler, names='value')
#display(nymap)
outfp = "choropleth_map4.html"
nymap1.save(outfp)
display(dropdown_year)              


# In[ ]:


display(output_year)


# In[ ]:





# # Finding focus contributing factors and vehicles
# The focus contributing factors are found on the basis of the 17 most common contributing factors in the pedestrian and the cyclist data frames. As the two data sets are not equal in size we take the 17 most commen contributing factors from both sets, however as most of these factors overlap we end up with 20 contributing factors.
# 
# The focus involved vehicles are found on the basis of the 18 most common vehicles involved in the pedestrian and the cyclist data frames. For the same reason as before we take the 18 most commen contributing factors from both sets, however as most of these factors overlap we end up with 20 contributing factors.

# In[87]:


data = pd.read_csv('data_preparred.csv',low_memory=False)
data['datetime']=pd.to_datetime(data['CRASH DATE'] + ' ' + data['CRASH TIME'])
filterdf = data[(pd.to_datetime(data['datetime']) >= '01/01/2016') & (pd.to_datetime(data['datetime']) <= '31/12/2020')]


# In[88]:


most_common_cyclist_causes = pd.concat([df_cyclist.iloc[:,18],df_cyclist.iloc[:,19],df_cyclist.iloc[:,20],df_cyclist.iloc[:,21],df_cyclist.iloc[:,22]])
most_common_cyclist_causes = most_common_cyclist_causes.loc[(most_common_cyclist_causes != 'Unspecified') & most_common_cyclist_causes.notna()]
most_common_pedestrian_causes = pd.concat([df_pedestrian.iloc[:,18],df_pedestrian.iloc[:,19],df_pedestrian.iloc[:,20],df_pedestrian.iloc[:,21],df_pedestrian.iloc[:,22]])
most_common_pedestrian_causes = most_common_pedestrian_causes.loc[(most_common_pedestrian_causes != 'Unspecified') & most_common_pedestrian_causes.notna()]


# In[89]:


# most common for cyclist
Counter(most_common_cyclist_causes).most_common(17)


# In[90]:


# most commen for pedestrian
Counter(most_common_pedestrian_causes).most_common(17)


# In[91]:


focus_causes = np.unique(list(dict(Counter(most_common_cyclist_causes).most_common(17)).keys()) +list(dict(Counter(most_common_pedestrian_causes).most_common(17)).keys())).tolist()
for i in focus_causes:
    print(i)


# In[118]:


most_common_cyclist_vehicles = pd.concat([df_cyclist.iloc[:,-9],df_cyclist.iloc[:,-8],df_cyclist.iloc[:,-7],df_cyclist.iloc[:,-6],df_cyclist.iloc[:,-5]])
most_common_cyclist_vehicles = most_common_cyclist_vehicles.loc[(most_common_cyclist_vehicles != 'UNKNOWN') & (most_common_cyclist_vehicles != 'OTHER') & most_common_cyclist_vehicles.notna()]
most_common_pedestrian_vehicles = pd.concat([df_pedestrian.iloc[:,-9],df_pedestrian.iloc[:,-8],df_pedestrian.iloc[:,-7],df_pedestrian.iloc[:,-6],df_pedestrian.iloc[:,-5]])
most_common_pedestrian_vehicles = most_common_pedestrian_vehicles.loc[(most_common_pedestrian_vehicles != 'UNKNOWN') & (most_common_pedestrian_vehicles != 'OTHER') & most_common_pedestrian_vehicles.notna()]
Counter(most_common_cyclist_vehicles).most_common(16)


# In[119]:


Counter(most_common_pedestrian_vehicles).most_common(15)


# In[121]:


focus_vehicles = np.unique(list(dict(Counter(most_common_pedestrian_vehicles).most_common()[1:19]).keys()) +list(dict(Counter(most_common_pedestrian_vehicles).most_common(18)).keys())).tolist()
for i in focus_vehicles:
    print(i)


# # Generating bokeh plots based on time for contributing factors

# In[122]:


df_hour_pedestrian = pd.DataFrame(0,index=np.arange(24),columns=[])
df_hour_cyclist = pd.DataFrame(0,index=np.arange(24),columns=[])
df_hourofweek_pedestrian = pd.DataFrame(0,index=np.arange(7*24),columns=[])
df_hourofweek_cyclist = pd.DataFrame(0,index=np.arange(7*24),columns=[])
df_day_pedestrian = pd.DataFrame(0,index=np.arange(7),columns=[])
df_day_cyclist = pd.DataFrame(0,index=np.arange(7),columns=[])
df_month_pedestrian = pd.DataFrame(0,index=np.arange(12)+1,columns=[])
df_month_cyclist = pd.DataFrame(0,index=np.arange(12)+1,columns=[])
df_year_pedestrian = pd.DataFrame(0,index=[2016,2017,2018,2019,2020],columns=[])
df_year_cyclist = pd.DataFrame(0,index=[2016,2017,2018,2019,2020],columns=[])
def by_hour(dataframe,cause):
    A = pd.DataFrame(0, index=np.arange(24),columns=[cause])
    AA = A
    B = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 1']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour]).count()['CONTRIBUTING FACTOR VEHICLE 1']
    C = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 2']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour]).count()['CONTRIBUTING FACTOR VEHICLE 2']
    D = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 3']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour]).count()['CONTRIBUTING FACTOR VEHICLE 3']
    E = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 4']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour]).count()['CONTRIBUTING FACTOR VEHICLE 4']
    F = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 5']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour]).count()['CONTRIBUTING FACTOR VEHICLE 5']
    AA = A.add(B,axis='index').fillna(0)
    AA = AA.add(A.add(B,axis='index').fillna(0))
    AA = AA.add(A.add(C,axis='index').fillna(0))
    AA = AA.add(A.add(D,axis='index').fillna(0))
    AA = AA.add(A.add(E,axis='index').fillna(0))
    AA = AA.add(A.add(F,axis='index').fillna(0)).astype(int)
    return AA

def by_hourofweek(dataframe,cause):
    A = pd.DataFrame(0, index=np.arange(24*7),columns=[cause])
    AA = A
    B = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 1']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour+24*pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 1']
    C = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 2']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour+24*pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 2']
    D = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 3']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour+24*pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 3']
    E = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 4']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour+24*pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 4']
    F = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 5']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour+24*pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 5']
    AA = A.add(B,axis='index').fillna(0)
    AA = AA.add(A.add(B,axis='index').fillna(0))
    AA = AA.add(A.add(C,axis='index').fillna(0))
    AA = AA.add(A.add(D,axis='index').fillna(0))
    AA = AA.add(A.add(E,axis='index').fillna(0))
    AA = AA.add(A.add(F,axis='index').fillna(0)).astype(int)
    return AA

def by_day(dataframe,cause):
    A = pd.DataFrame(0, index=np.arange(7),columns=[cause])
    AA = A
    B = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 1']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 1']
    C = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 2']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 2']
    D = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 3']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 3']
    E = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 4']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 4']
    F = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 5']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 5']
    AA = A.add(B,axis='index').fillna(0)
    AA = AA.add(A.add(B,axis='index').fillna(0))
    AA = AA.add(A.add(C,axis='index').fillna(0))
    AA = AA.add(A.add(D,axis='index').fillna(0))
    AA = AA.add(A.add(E,axis='index').fillna(0))
    AA = AA.add(A.add(F,axis='index').fillna(0))
    return AA

def by_month(dataframe,cause):
    A = pd.DataFrame(0, index=np.arange(12)+1,columns=[cause])
    AA = A
    B = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 1']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.month]).count()['CONTRIBUTING FACTOR VEHICLE 1']
    C = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 2']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.month]).count()['CONTRIBUTING FACTOR VEHICLE 2']
    D = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 3']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.month]).count()['CONTRIBUTING FACTOR VEHICLE 3']
    E = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 4']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.month]).count()['CONTRIBUTING FACTOR VEHICLE 4']
    F = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 5']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.month]).count()['CONTRIBUTING FACTOR VEHICLE 5']
    AA = A.add(B,axis='index').fillna(0)
    AA = AA.add(A.add(B,axis='index').fillna(0))
    AA = AA.add(A.add(C,axis='index').fillna(0))
    AA = AA.add(A.add(D,axis='index').fillna(0))
    AA = AA.add(A.add(E,axis='index').fillna(0))
    AA = AA.add(A.add(F,axis='index').fillna(0)).astype(int)
    return AA

def by_year(dataframe,cause):
    A = pd.DataFrame(0, index=np.array([2016,2017,2018,2019,2020]),columns=[cause])
    AA = A
    B = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 1']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.year]).count()['CONTRIBUTING FACTOR VEHICLE 1']
    C = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 2']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.year]).count()['CONTRIBUTING FACTOR VEHICLE 2']
    D = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 3']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.year]).count()['CONTRIBUTING FACTOR VEHICLE 3']
    E = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 4']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.year]).count()['CONTRIBUTING FACTOR VEHICLE 4']
    F = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 5']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.year]).count()['CONTRIBUTING FACTOR VEHICLE 5']
    AA = A.add(B,axis='index').fillna(0)
    AA = AA.add(A.add(B,axis='index').fillna(0))
    AA = AA.add(A.add(C,axis='index').fillna(0))
    AA = AA.add(A.add(D,axis='index').fillna(0))
    AA = AA.add(A.add(E,axis='index').fillna(0))
    AA = AA.add(A.add(F,axis='index').fillna(0)).astype(int)
    return AA

for k in range(0,len(list(focus_causes))):
    df_hour_pedestrian[focus_causes[k]] = by_hour(df_pedestrian,focus_causes[k])
    df_hour_cyclist[focus_causes[k]] = by_hour(df_cyclist,focus_causes[k])
    df_day_pedestrian[focus_causes[k]] = by_day(df_pedestrian,focus_causes[k])
    df_day_cyclist[focus_causes[k]] = by_day(df_cyclist,focus_causes[k])
    df_month_pedestrian[focus_causes[k]] = by_month(df_pedestrian,focus_causes[k])
    df_month_cyclist[focus_causes[k]] = by_month(df_cyclist,focus_causes[k])
    df_year_pedestrian[focus_causes[k]] = by_year(df_pedestrian,focus_causes[k])
    df_year_cyclist[focus_causes[k]] = by_year(df_cyclist,focus_causes[k])
    df_hourofweek_pedestrian[focus_causes[k]] = by_hourofweek(df_pedestrian,focus_causes[k])
    df_hourofweek_cyclist[focus_causes[k]] = by_hourofweek(df_cyclist,focus_causes[k])
df_hour_pedestrian.index.rename('Time',inplace=True)
df_hour_cyclist.index.rename('Time',inplace=True)
df_day_pedestrian.index.rename('Time',inplace=True)
df_day_cyclist.index.rename('Time',inplace=True)
df_day_pedestrian = df_day_pedestrian.astype(int)
df_day_cyclist = df_day_cyclist.astype(int)
df_hourofweek_pedestrian.index.rename('Time',inplace=True)
df_hourofweek_cyclist.index.rename('Time',inplace=True)
df_month_pedestrian.index.rename('Time',inplace=True)
df_month_cyclist.index.rename('Time',inplace=True)
df_year_pedestrian.index.rename('Time',inplace=True)
df_year_cyclist.index.rename('Time',inplace=True)


# In[255]:


focus_causes1 = copy.deepcopy(focus_causes)
focus_causes1[13] = 'Pedestrian/Bicyclist Error/Confusion'
# renaming of category 'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion' to 'Pedestrian/Bicyclist Error/Confusion'
# to make legends smaller


# In[124]:


df11 = df_hour_pedestrian.copy()
df11.index = df_hour_pedestrian.index.astype(str)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
for j in range(0,np.shape(df_hour_pedestrian)[1]):
    tot = sum(df_hour_pedestrian.iloc[:,j])
    for i in range(0,np.shape(df_hour_pedestrian)[0]):
        df11.iloc[i,j] = df_hour_pedestrian.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(0, 23, num = 24)]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_causes1):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Contributing factors for pedestrian accidents by hour of the day'
p.xaxis.axis_label='Hour of the day'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#p.legend.oversize_policy = "popup"
#p.legend.label_text_font_size = "10px"
#p.legend.orientation = "horizontal"
#show(p)
p.legend.label_text_font_size = "10px"
output_file("hour_pedestrian_causes.html")
save(p)


# In[125]:


df11 = df_hour_cyclist.copy()
df11.index = (df_hour_cyclist.index).astype(str)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
for j in range(0,np.shape(df_hour_cyclist)[1]):
    tot = sum(df_hour_cyclist.iloc[:,j])
    for i in range(0,np.shape(df_hour_cyclist)[0]):
        df11.iloc[i,j] = df_hour_cyclist.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(0, 23, num = 24)]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_causes1):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Contributing factors for cyclist accidents per hour'
p.xaxis.axis_label='Hour of the day'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#show(p)
p.legend.label_text_font_size = "10px"
output_file("hour_cyclist_causes.html")
save(p)


# In[126]:


df11 = df_month_pedestrian.copy()
df11.index = df_month_pedestrian.index.astype(str)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
for j in range(0,np.shape(df_month_pedestrian)[1]):
    tot = sum(df_month_pedestrian.iloc[:,j])
    for i in range(0,np.shape(df_month_pedestrian)[0]):
        df11.iloc[i,j] = df_month_pedestrian.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(1, 12, num = 12)]
source = ColumnDataSource(df11)
p = figure(x_range= FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_causes1):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Contributing factors for pedestrian accidents per month'
p.xaxis.axis_label='Month of the year'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#show(p)
p.legend.label_text_font_size = "10px"
output_file("month_pedestrian_causes.html")
save(p)


# In[127]:


df11 = df_month_cyclist.copy()
df11.index = df_month_cyclist.index.astype(str)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)

for j in range(0,np.shape(df_month_cyclist)[1]):
    tot = sum(df_month_cyclist.iloc[:,j])
    for i in range(0,np.shape(df_month_cyclist)[0]):
        df11.iloc[i,j] = df_month_cyclist.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(1, 12, num = 12)]
source = ColumnDataSource(df11)
p = figure(x_range= FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_causes1):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Contributing factors for cyclist accidents per month'
p.xaxis.axis_label='Month of the year'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
p.legend.label_text_font_size = "10px"
output_file("month_cyclist_causes.html")
save(p)


# In[128]:


df11 = df_year_pedestrian.copy()
df11.index = df_year_pedestrian.index.astype(str)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
for j in range(0,np.shape(df_year_pedestrian)[1]):
    tot = sum(df_year_pedestrian.iloc[:,j])
    for i in range(0,np.shape(df_year_pedestrian)[0]):
        df11.iloc[i,j] = df_year_pedestrian.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(2016, 2020, num = 5)]
source = ColumnDataSource(df11)
p = figure(x_range= FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_causes1):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Contributing factors for pedestrian accidents per month'
p.xaxis.axis_label='Year'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#show(p)
p.legend.label_text_font_size = "10px"
output_file("year_pedestrian_causes.html")
save(p)


# In[129]:


df11 = df_year_cyclist.copy()
df11.index = df_year_cyclist.index.astype(str)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
for j in range(0,np.shape(df_year_cyclist)[1]):
    tot = sum(df_year_cyclist.iloc[:,j])
    for i in range(0,np.shape(df_year_cyclist)[0]):
        df11.iloc[i,j] = df_year_cyclist.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(2016, 2020, num = 5)]
source = ColumnDataSource(df11)
p = figure(x_range= FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_causes1):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=0.8, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Contributing factors for cyclist accidents per month'
p.xaxis.axis_label='Year'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#show(p)
p.legend.label_text_font_size = "10px"
output_file("year_cyclist_causes.html")
save(p)


# In[130]:


df11 = df_day_pedestrian.copy()
df11.index = (df_day_pedestrian.index+1).astype(str)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
for j in range(0,np.shape(df_day_pedestrian)[1]):
    tot = sum(df_day_pedestrian.iloc[:,j])
    for i in range(0,np.shape(df_day_pedestrian)[0]):
        df11.iloc[i,j] = df_day_pedestrian.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(1, 7, num = 7)]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_causes1):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Contributing factors for pedestrian accidents per weekday'
p.xaxis.axis_label='Day of the week'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#show(p)
p.legend.label_text_font_size = "10px"
output_file("weekday_pedestrian_causes.html")
save(p)


# In[131]:


df11 = df_day_cyclist.copy()
df11.index = (df_day_cyclist.index+1).astype(str)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
for j in range(0,np.shape(df_day_cyclist)[1]):
    tot = sum(df_day_cyclist.iloc[:,j])
    for i in range(0,np.shape(df_day_cyclist)[0]):
        df11.iloc[i,j] = df_day_cyclist.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(1, 7, num = 7)]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_causes1):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Contributing factors for cyclist accidents per weekday'
p.xaxis.axis_label='Day of the week'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#show(p)
p.legend.label_text_font_size = "10px"
output_file("weekday_cyclist_causes.html")
save(p)


# In[132]:


df11 = df_hourofweek_cyclist.copy()
df11.index = (df_hourofweek_cyclist.index).astype(str)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
for j in range(0,np.shape(df_hourofweek_cyclist)[1]):
    tot = sum(df_hourofweek_cyclist.iloc[:,j])
    for i in range(0,np.shape(df_hourofweek_cyclist)[0]):
        df11.iloc[i,j] = df_hourofweek_cyclist.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(0, 167, num = 168)]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=2000)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_causes1):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Contributing factors for cyclist accidents per hour of week'
p.xaxis.axis_label='Day of the week'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#show(p)


# # Causes of accidents in relation to weather

# In[195]:


df3['Weather type'].unique()
DRY_ROAD_PLUS = np.zeros(len(focus_causes))
DRY_ROAD_MINUS = np.zeros(len(focus_causes))
HEAVY_WET_ROAD = np.zeros(len(focus_causes))
LIGHT_WET_ROAD = np.zeros(len(focus_causes))
LIGHT_SNOWY_ROAD = np.zeros(len(focus_causes))
HEAVY_SNOWY_ROAD = np.zeros(len(focus_causes))
df_dry_plus = df3.loc[df3['Weather type'] == 'Dry road+']
df_dry_minus = df3.loc[df3['Weather type'] == 'Dry road-']
df_heavy_wet = df3.loc[df3['Weather type'] == 'Heavy wet road']
df_light_wet = df3.loc[df3['Weather type'] == 'Ligth wet road']
df_light_snowy = df3.loc[df3['Weather type'] == 'Ligth snowy road']
df_heavy_snowy = df3.loc[df3['Weather type'] == 'Heavy snowy road']

for i in range(0,len(focus_causes)):
    summen = 0 
    summen = sum(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    DRY_ROAD_PLUS[i] = summen
    summen = 0 
    summen = sum(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    DRY_ROAD_MINUS[i] = summen
    summen = 0 
    summen = sum(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    HEAVY_WET_ROAD[i] = summen
    summen = 0 
    summen = sum(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_light_wet['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_light_wet['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_light_wet['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_light_wet['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    LIGHT_WET_ROAD[i] = summen
    summen = 0 
    summen = sum(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_light_snowy['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_light_snowy['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_light_snowy['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_light_snowy['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    LIGHT_SNOWY_ROAD[i] = summen
    summen = 0 
    summen = sum(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    HEAVY_SNOWY_ROAD[i] = summen
    
#df3['Weather type'].unique()
#[i] = summen

#df3['Weather type'].unique()


# In[236]:


df_rel = pd.DataFrame()
total_dry_plus = df_dry_plus.iloc[:,29:34].notna().astype('uint8').sum().sum()
total_dry_minus = df_dry_minus.iloc[:,29:34].notna().astype('uint8').sum().sum()
total_light_wet = df_light_wet.iloc[:,29:34].notna().astype('uint8').sum().sum()
total_heavy_wet = df_heavy_wet.iloc[:,29:34].notna().astype('uint8').sum().sum()
total_light_snowy = df_light_snowy.iloc[:,29:34].notna().astype('uint8').sum().sum()
total_heavy_snowy = df_heavy_snowy.iloc[:,29:34].notna().astype('uint8').sum().sum()
#total_pedestrian = df_pedestrian.iloc[:,18:23].notna().astype('uint8').sum().sum()
#total_cyclist = df_cyclist.iloc[:,18:23].notna().astype('uint8').sum().sum()
df_rel['Name'] = focus_causes
df_rel['Dry plus'] = np.zeros(len(focus_causes))
df_rel['Dry minus'] = np.zeros(len(focus_causes))
df_rel['Light snow'] = np.zeros(len(focus_causes))
df_rel['Heavy snow'] = np.zeros(len(focus_causes))
df_rel['Light rain'] = np.zeros(len(focus_causes))
df_rel['Heavy rain'] = np.zeros(len(focus_causes))

for i in range(0,len(focus_causes)):
    df_rel['Dry plus'].iloc[i] = (sum((df_dry_plus['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_dry_plus['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/len(df_dry_plus)
    df_rel['Dry minus'].iloc[i] = (sum((df_dry_minus['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_dry_minus['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/len(df_dry_minus)
    df_rel['Light snow'].iloc[i] = (sum((df_light_snowy['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_light_snowy['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_light_snowy['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_light_snowy['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_light_snowy['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/len(df_light_snowy)
    df_rel['Heavy snow'].iloc[i] = (sum((df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/len(df_heavy_snowy)
    df_rel['Light rain'].iloc[i] = (sum((df_light_wet['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_light_wet['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_light_wet['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_light_wet['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_light_wet['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/len(df_light_wet)
    df_rel['Heavy rain'].iloc[i] = (sum((df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/len(df_heavy_wet)
                                 
df_rel.set_index('Name',inplace=True)
df_rel.index.name = None
df_rel


# In[241]:


df_rel.index.rename('Time',inplace=True)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
df11 = df_rel.copy()

hours = focus_causes1#[str(round(x)) for x in np.linspace(0, len(focus_causes)-1, num = len(focus_causes))]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(df_rel.columns):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category10[6][indx])

p.title.text='Average daily contributing factors for weather type'
#p.xaxis.axis_label='Hour of the day'
p.yaxis.axis_label='Average daily contributing factors for weater type'
p.legend.orientation = "vertical"
p.xaxis.major_label_orientation = np.pi/3
#p.xaxis.major_label_orientation = "vertical"
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#p.yaxis.formatter = NumeralTickFormatter(format='0 %')
#p.legend.orientation = "horizontal"
p.legend.label_text_font_size = "10px"

output_file("avg_causes_frequency_weather.html")
save(p)
#show(p)


# In[243]:


df_rel = pd.DataFrame()
total_dry_plus = df_dry_plus.iloc[:,29:34].notna().astype('uint8').sum().sum()
total_dry_minus = df_dry_minus.iloc[:,29:34].notna().astype('uint8').sum().sum()
total_light_wet = df_light_wet.iloc[:,29:34].notna().astype('uint8').sum().sum()
total_heavy_wet = df_heavy_wet.iloc[:,29:34].notna().astype('uint8').sum().sum()
total_light_snowy = df_light_snowy.iloc[:,29:34].notna().astype('uint8').sum().sum()
total_heavy_snowy = df_heavy_snowy.iloc[:,29:34].notna().astype('uint8').sum().sum()
#total_pedestrian = df_pedestrian.iloc[:,18:23].notna().astype('uint8').sum().sum()
#total_cyclist = df_cyclist.iloc[:,18:23].notna().astype('uint8').sum().sum()
df_rel['Name'] = focus_causes
df_rel['Dry plus'] = np.zeros(len(focus_causes))
df_rel['Dry minus'] = np.zeros(len(focus_causes))
df_rel['Light snow'] = np.zeros(len(focus_causes))
df_rel['Heavy snow'] = np.zeros(len(focus_causes))
df_rel['Light rain'] = np.zeros(len(focus_causes))
df_rel['Heavy rain'] = np.zeros(len(focus_causes))

for i in range(0,len(focus_causes)):
    df_rel['Dry plus'].iloc[i] = (sum((df_dry_plus['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_dry_plus['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_dry_plus['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/total_dry_plus
    df_rel['Dry minus'].iloc[i] = (sum((df_dry_minus['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_dry_minus['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_dry_minus['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/total_dry_minus
    df_rel['Light snow'].iloc[i] = (sum((df_light_snowy['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_light_snowy['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_light_snowy['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_light_snowy['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_light_snowy['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/total_light_snowy
    df_rel['Heavy snow'].iloc[i] = (sum((df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_heavy_snowy['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/total_heavy_snowy
    df_rel['Light rain'].iloc[i] = (sum((df_light_wet['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_light_wet['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_light_wet['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_light_wet['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_light_wet['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/total_light_wet
    df_rel['Heavy rain'].iloc[i] = (sum((df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_heavy_wet['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/total_heavy_wet
                                 
df_rel.set_index('Name',inplace=True)
df_rel.index.name = None
#df_rel


# In[247]:


df_rel.index.rename('Time',inplace=True)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
df11 = df_rel.copy()

hours = focus_causes1#[str(round(x)) for x in np.linspace(0, len(focus_causes)-1, num = len(focus_causes))]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(df_rel.columns):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category10[6][indx])

p.title.text='Frequency daily contributing factors for weather type'
#p.xaxis.axis_label='Hour of the day'
p.yaxis.axis_label='Frequency contributing factors for weater type'
p.legend.orientation = "vertical"
p.xaxis.major_label_orientation = np.pi/3
#p.xaxis.major_label_orientation = "vertical"
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
p.yaxis.formatter = NumeralTickFormatter(format='0 %')
#p.legend.orientation = "horizontal"
p.legend.label_text_font_size = "10px"

output_file("percent_causes_frequency_weather.html")
save(p)
#show(p)


# # Generating bokeh plots based on time for focus vehicles

# In[133]:


df_hour_pedestrian = pd.DataFrame(0,index=np.arange(24),columns=[])
df_hour_cyclist = pd.DataFrame(0,index=np.arange(24),columns=[])
df_hourofweek_pedestrian = pd.DataFrame(0,index=np.arange(7*24),columns=[])
df_hourofweek_cyclist = pd.DataFrame(0,index=np.arange(7*24),columns=[])
df_day_pedestrian = pd.DataFrame(0,index=np.arange(7),columns=[])
df_day_cyclist = pd.DataFrame(0,index=np.arange(7),columns=[])
df_month_pedestrian = pd.DataFrame(0,index=np.arange(12)+1,columns=[])
df_month_cyclist = pd.DataFrame(0,index=np.arange(12)+1,columns=[])
df_year_pedestrian = pd.DataFrame(0,index=[2016,2017,2018,2019,2020],columns=[])
df_year_cyclist = pd.DataFrame(0,index=[2016,2017,2018,2019,2020],columns=[])
def by_hour(dataframe,cause):
    A = pd.DataFrame(0, index=np.arange(24),columns=[cause])
    AA = A
    B = dataframe.loc[dataframe['VEHICLE TYPE CODE 1']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour]).count()['VEHICLE TYPE CODE 1']
    C = dataframe.loc[dataframe['VEHICLE TYPE CODE 2']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour]).count()['VEHICLE TYPE CODE 2']
    D = dataframe.loc[dataframe['VEHICLE TYPE CODE 3']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour]).count()['VEHICLE TYPE CODE 3']
    E = dataframe.loc[dataframe['VEHICLE TYPE CODE 4']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour]).count()['VEHICLE TYPE CODE 4']
    F = dataframe.loc[dataframe['VEHICLE TYPE CODE 5']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour]).count()['VEHICLE TYPE CODE 5']
    AA = A.add(B,axis='index').fillna(0)
    AA = AA.add(A.add(B,axis='index').fillna(0))
    AA = AA.add(A.add(C,axis='index').fillna(0))
    AA = AA.add(A.add(D,axis='index').fillna(0))
    AA = AA.add(A.add(E,axis='index').fillna(0))
    AA = AA.add(A.add(F,axis='index').fillna(0)).astype(int)
    return AA

def by_hourofweek(dataframe,cause):
    A = pd.DataFrame(0, index=np.arange(24*7),columns=[cause])
    AA = A
    B = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 1']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour+24*pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 1']
    C = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 2']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour+24*pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 2']
    D = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 3']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour+24*pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 3']
    E = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 4']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour+24*pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 4']
    F = dataframe.loc[dataframe['CONTRIBUTING FACTOR VEHICLE 5']==cause].groupby([pd.to_datetime(dataframe['CRASH TIME'],format = "%H:%M").dt.hour+24*pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['CONTRIBUTING FACTOR VEHICLE 5']
    AA = A.add(B,axis='index').fillna(0)
    AA = AA.add(A.add(B,axis='index').fillna(0))
    AA = AA.add(A.add(C,axis='index').fillna(0))
    AA = AA.add(A.add(D,axis='index').fillna(0))
    AA = AA.add(A.add(E,axis='index').fillna(0))
    AA = AA.add(A.add(F,axis='index').fillna(0)).astype(int)
    return AA

def by_day(dataframe,cause):
    A = pd.DataFrame(0, index=np.arange(7),columns=[cause])
    AA = A
    B = dataframe.loc[dataframe['VEHICLE TYPE CODE 1']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['VEHICLE TYPE CODE 1']
    C = dataframe.loc[dataframe['VEHICLE TYPE CODE 2']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['VEHICLE TYPE CODE 2']
    D = dataframe.loc[dataframe['VEHICLE TYPE CODE 3']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['VEHICLE TYPE CODE 3']
    E = dataframe.loc[dataframe['VEHICLE TYPE CODE 4']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['VEHICLE TYPE CODE 4']
    F = dataframe.loc[dataframe['VEHICLE TYPE CODE 5']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.weekday]).count()['VEHICLE TYPE CODE 5']
    AA = A.add(B,axis='index').fillna(0)
    AA = AA.add(A.add(B,axis='index').fillna(0))
    AA = AA.add(A.add(C,axis='index').fillna(0))
    AA = AA.add(A.add(D,axis='index').fillna(0))
    AA = AA.add(A.add(E,axis='index').fillna(0))
    AA = AA.add(A.add(F,axis='index').fillna(0))
    return AA

def by_month(dataframe,cause):
    A = pd.DataFrame(0, index=np.arange(12)+1,columns=[cause])
    AA = A
    B = dataframe.loc[dataframe['VEHICLE TYPE CODE 1']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.month]).count()['VEHICLE TYPE CODE 1']
    C = dataframe.loc[dataframe['VEHICLE TYPE CODE 2']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.month]).count()['VEHICLE TYPE CODE 2']
    D = dataframe.loc[dataframe['VEHICLE TYPE CODE 3']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.month]).count()['VEHICLE TYPE CODE 3']
    E = dataframe.loc[dataframe['VEHICLE TYPE CODE 4']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.month]).count()['VEHICLE TYPE CODE 4']
    F = dataframe.loc[dataframe['VEHICLE TYPE CODE 5']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.month]).count()['VEHICLE TYPE CODE 5']
    AA = A.add(B,axis='index').fillna(0)
    AA = AA.add(A.add(B,axis='index').fillna(0))
    AA = AA.add(A.add(C,axis='index').fillna(0))
    AA = AA.add(A.add(D,axis='index').fillna(0))
    AA = AA.add(A.add(E,axis='index').fillna(0))
    AA = AA.add(A.add(F,axis='index').fillna(0)).astype(int)
    return AA

def by_year(dataframe,cause):
    A = pd.DataFrame(0, index=np.array([2016,2017,2018,2019,2020]),columns=[cause])
    AA = A
    B = dataframe.loc[dataframe['VEHICLE TYPE CODE 1']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.year]).count()['VEHICLE TYPE CODE 1']
    C = dataframe.loc[dataframe['VEHICLE TYPE CODE 2']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.year]).count()['VEHICLE TYPE CODE 2']
    D = dataframe.loc[dataframe['VEHICLE TYPE CODE 3']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.year]).count()['VEHICLE TYPE CODE 3']
    E = dataframe.loc[dataframe['VEHICLE TYPE CODE 4']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.year]).count()['VEHICLE TYPE CODE 4']
    F = dataframe.loc[dataframe['VEHICLE TYPE CODE 5']==cause].groupby([pd.to_datetime(dataframe['CRASH DATE'],format = '%m/%d/%Y').dt.year]).count()['VEHICLE TYPE CODE 5']
    AA = A.add(B,axis='index').fillna(0)
    AA = AA.add(A.add(B,axis='index').fillna(0))
    AA = AA.add(A.add(C,axis='index').fillna(0))
    AA = AA.add(A.add(D,axis='index').fillna(0))
    AA = AA.add(A.add(E,axis='index').fillna(0))
    AA = AA.add(A.add(F,axis='index').fillna(0)).astype(int)
    return AA

for k in range(0,len(list(focus_vehicles))):
    df_hour_pedestrian[focus_vehicles[k]] = by_hour(df_pedestrian,focus_vehicles[k])
    df_hour_cyclist[focus_vehicles[k]] = by_hour(df_cyclist,focus_vehicles[k])
    df_day_pedestrian[focus_vehicles[k]] = by_day(df_pedestrian,focus_vehicles[k])
    df_day_cyclist[focus_vehicles[k]] = by_day(df_cyclist,focus_vehicles[k])
    df_month_pedestrian[focus_vehicles[k]] = by_month(df_pedestrian,focus_vehicles[k])
    df_month_cyclist[focus_vehicles[k]] = by_month(df_cyclist,focus_vehicles[k])
    df_year_pedestrian[focus_vehicles[k]] = by_year(df_pedestrian,focus_vehicles[k])
    df_year_cyclist[focus_vehicles[k]] = by_year(df_cyclist,focus_vehicles[k])
    df_hourofweek_pedestrian[focus_vehicles[k]] = by_hourofweek(df_pedestrian,focus_vehicles[k])
    df_hourofweek_cyclist[focus_vehicles[k]] = by_hourofweek(df_cyclist,focus_vehicles[k])
df_hour_pedestrian.index.rename('Time',inplace=True)
df_hour_cyclist.index.rename('Time',inplace=True)
df_day_pedestrian.index.rename('Time',inplace=True)
df_day_cyclist.index.rename('Time',inplace=True)
df_day_pedestrian = df_day_pedestrian.astype(int)
df_day_cyclist = df_day_cyclist.astype(int)
df_hourofweek_pedestrian.index.rename('Time',inplace=True)
df_hourofweek_cyclist.index.rename('Time',inplace=True)
df_month_pedestrian.index.rename('Time',inplace=True)
df_month_cyclist.index.rename('Time',inplace=True)
df_year_pedestrian.index.rename('Time',inplace=True)
df_year_cyclist.index.rename('Time',inplace=True)


# In[134]:


focus_vehicles


# In[135]:


df11 = df_hour_pedestrian.copy()
df11.index = (df_hour_pedestrian.index+1).astype(str)
for j in range(0,np.shape(df_hour_pedestrian)[1]):
    tot = sum(df_hour_pedestrian.iloc[:,j])
    for i in range(0,np.shape(df_hour_pedestrian)[0]):
        df11.iloc[i,j] = df_hour_cyclist.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(0, 23, num = 24)]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_vehicles):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Rate of vehicle involved in pedestrian accidents by hour of the day'
p.xaxis.axis_label='Hour of the day'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#p.legend.oversize_policy = "popup"
#p.legend.label_text_font_size = "10px"
#p.legend.orientation = "horizontal"
#show(p)
p.legend.label_text_font_size = "10px"
output_file("hour_pedestrian_vehicles.html")
save(p)


# In[136]:


df11 = df_hour_cyclist.copy()
df11.index = (df_hour_cyclist.index+1).astype(str)
for j in range(0,np.shape(df_hour_cyclist)[1]):
    tot = sum(df_hour_cyclist.iloc[:,j])
    for i in range(0,np.shape(df_hour_cyclist)[0]):
        df11.iloc[i,j] = df_hour_cyclist.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(0, 23, num = 24)]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_vehicles):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Rate of vehicle involved in cyclist accidents by hour of the day'
p.xaxis.axis_label='Hour of the day'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#p.legend.oversize_policy = "popup"
#p.legend.label_text_font_size = "10px"
#p.legend.orientation = "horizontal"
#show(p)
p.legend.label_text_font_size = "10px"
output_file("hour_pedestrian_vehicles.html")
save(p)


# In[137]:


df11 = df_day_cyclist.copy()
df11.index = (df_day_cyclist.index+1).astype(str)
for j in range(0,np.shape(df_day_cyclist)[1]):
    tot = sum(df_day_cyclist.iloc[:,j])
    for i in range(0,np.shape(df_day_cyclist)[0]):
        df11.iloc[i,j] = df_day_cyclist.iloc[i,j]/tot
hours = [str(round(x)) for x in np.linspace(1, 7, num = 7)]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(focus_vehicles):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category20[len(focus_causes)][indx])

p.title.text='Contributing factors for pedestrian accidents per weekday'
p.xaxis.axis_label='Day of the week'
p.yaxis.axis_label='Relative frequency'
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
#show(p)
p.legend.label_text_font_size = "10px"
output_file("weekday_cyclist_vehicles.html")
save(p)


# # Generating distribution plots (fatal vs injury vs no injury)

# In[138]:


non_fatal_cyclist = np.zeros(len(focus_causes))
injury_cyclist = np.zeros(len(focus_causes))
killed_cyclist = np.zeros(len(focus_causes))
df_nonfatal_cyclist = df_cyclist.loc[df_cyclist['NUMBER OF PEDESTRIANS KILLED']+df_cyclist['NUMBER OF PEDESTRIANS INJURED']+df_cyclist['NUMBER OF CYCLIST KILLED']+df_cyclist['NUMBER OF CYCLIST INJURED']==0]
df_injury_cyclist = df_cyclist.loc[(df_cyclist['NUMBER OF PEDESTRIANS INJURED']+df_cyclist['NUMBER OF CYCLIST INJURED']>0) & (df_cyclist['NUMBER OF PEDESTRIANS KILLED'] + df_cyclist['NUMBER OF CYCLIST KILLED'] == 0)]
df_killed_cyclist = df_cyclist.loc[df_cyclist['NUMBER OF PEDESTRIANS KILLED']+df_cyclist['NUMBER OF CYCLIST KILLED']>0]
for i in range(0,len(focus_causes)):
    summen = 0 
    summen = sum(df_nonfatal_cyclist['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_nonfatal_cyclist['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_nonfatal_cyclist['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_nonfatal_cyclist['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_nonfatal_cyclist['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    non_fatal_cyclist[i] = summen
    summen = 0 
    summen = sum(df_injury_cyclist['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_injury_cyclist['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_injury_cyclist['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_injury_cyclist['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_injury_cyclist['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    injury_cyclist[i] = summen
    summen = 0 
    summen = sum(df_killed_cyclist['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_killed_cyclist['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_killed_cyclist['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_killed_cyclist['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_killed_cyclist['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    killed_cyclist[i] = summen


# In[139]:


from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter
df_distributions = pd.DataFrame()
df_distributions['Name'] = focus_causes
df_distributions['No injuries'] = non_fatal_cyclist.astype(int)
df_distributions['With injuries'] = injury_cyclist.astype(int)
df_distributions['Fatal'] = killed_cyclist.astype(int)
df_distributions

df_total = df_distributions["No injuries"] + df_distributions["With injuries"] + df_distributions["Fatal"]
df_rel = df_distributions[df_distributions.columns[1:]].div(df_total, 0)

years = ["No injuries", "With injuries", "Fatal"]
colors = list(Category20[20][0:3])

data1 = {'Causes' : focus_causes,
        'No injuries'   : list(df_rel['No injuries']),
        'With injuries'   : list(df_rel['With injuries']),
        'Fatal'   : list(df_rel['Fatal'])}

p = figure(x_range=focus_causes, plot_width=1200,plot_height=1000, title="Causes and distributions of severity for accidents involving cyclists",
           toolbar_location=None, tools="")

p.vbar_stack(years, x='Causes', width=0.9, color=colors, source=data1,
             legend_label=years)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "center_left"
p.legend.orientation = "vertical"
p.xaxis.major_label_orientation = "vertical"
p.yaxis.formatter = NumeralTickFormatter(format='0 %')
p.legend.label_text_font_size = "10px"
output_file("distibution_causes_cyclist.html")
save(p)


# In[140]:


non_fatal_cyclist = np.zeros(len(focus_vehicles))
injury_cyclist = np.zeros(len(focus_vehicles))
killed_cyclist = np.zeros(len(focus_vehicles))
df_nonfatal_cyclist = df_cyclist.loc[df_cyclist['NUMBER OF PEDESTRIANS KILLED']+df_cyclist['NUMBER OF PEDESTRIANS INJURED']+df_cyclist['NUMBER OF CYCLIST KILLED']+df_cyclist['NUMBER OF CYCLIST INJURED']==0]
df_injury_cyclist = df_cyclist.loc[(df_cyclist['NUMBER OF PEDESTRIANS INJURED']+df_cyclist['NUMBER OF CYCLIST INJURED']>0) & (df_cyclist['NUMBER OF PEDESTRIANS KILLED'] + df_cyclist['NUMBER OF CYCLIST KILLED'] == 0)]
df_killed_cyclist = df_cyclist.loc[df_cyclist['NUMBER OF PEDESTRIANS KILLED']+df_cyclist['NUMBER OF CYCLIST KILLED']>0]
for i in range(0,len(focus_vehicles)):
    summen = 0 
    summen = sum(df_nonfatal_cyclist['VEHICLE TYPE CODE 1'] == focus_vehicles[i])
    summen = summen + sum(df_nonfatal_cyclist['VEHICLE TYPE CODE 2'] == focus_vehicles[i])
    summen = summen + sum(df_nonfatal_cyclist['VEHICLE TYPE CODE 3'] == focus_vehicles[i])
    summen = summen + sum(df_nonfatal_cyclist['VEHICLE TYPE CODE 4'] == focus_vehicles[i])
    summen = summen + sum(df_nonfatal_cyclist['VEHICLE TYPE CODE 5'] == focus_vehicles[i])
    non_fatal_cyclist[i] = summen
    summen = 0 
    summen = sum(df_injury_cyclist['VEHICLE TYPE CODE 1'] == focus_vehicles[i])
    summen = summen + sum(df_injury_cyclist['VEHICLE TYPE CODE 2'] == focus_vehicles[i])
    summen = summen + sum(df_injury_cyclist['VEHICLE TYPE CODE 3'] == focus_vehicles[i])
    summen = summen + sum(df_injury_cyclist['VEHICLE TYPE CODE 4'] == focus_vehicles[i])
    summen = summen + sum(df_injury_cyclist['VEHICLE TYPE CODE 5'] == focus_vehicles[i])
    injury_cyclist[i] = summen
    summen = 0 
    summen = sum(df_killed_cyclist['VEHICLE TYPE CODE 1'] == focus_vehicles[i])
    summen = summen + sum(df_killed_cyclist['VEHICLE TYPE CODE 2'] == focus_vehicles[i])
    summen = summen + sum(df_killed_cyclist['VEHICLE TYPE CODE 3'] == focus_vehicles[i])
    summen = summen + sum(df_killed_cyclist['VEHICLE TYPE CODE 4'] == focus_vehicles[i])
    summen = summen + sum(df_killed_cyclist['VEHICLE TYPE CODE 5'] == focus_vehicles[i])
    killed_cyclist[i] = summen


# In[141]:


from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter
df_distributions = pd.DataFrame()
df_distributions['Name'] = focus_vehicles
df_distributions['No injuries'] = non_fatal_cyclist.astype(int)
df_distributions['With injuries'] = injury_cyclist.astype(int)
df_distributions['Fatal'] = killed_cyclist.astype(int)
df_distributions

df_total = df_distributions["No injuries"] + df_distributions["With injuries"] + df_distributions["Fatal"]
df_rel = df_distributions[df_distributions.columns[1:]].div(df_total, 0)

years = ["No injuries", "With injuries", "Fatal"]
colors = list(Category20[19][0:3])

data1 = {'Causes' : focus_vehicles,
        'No injuries'   : list(df_rel['No injuries']),
        'With injuries'   : list(df_rel['With injuries']),
        'Fatal'   : list(df_rel['Fatal'])}

p = figure(x_range=focus_vehicles, plot_width=1200,plot_height=1000, title="Vehicle types and distributions of severity for accidents involving cyclists",
           toolbar_location=None, tools="")

p.vbar_stack(years, x='Causes', width=0.9, color=colors, source=data1,
             legend_label=years)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "center_left"
p.legend.orientation = "vertical"
p.xaxis.major_label_orientation = "vertical"
p.yaxis.formatter = NumeralTickFormatter(format='0 %')
p.legend.label_text_font_size = "10px"
output_file("distibution_vehicles_cyclist.html")
save(p)


# In[142]:


non_fatal_pedestrian = np.zeros(len(focus_causes))
injury_pedestrian = np.zeros(len(focus_causes))
killed_pedestrian = np.zeros(len(focus_causes))
df_nonfatal_pedestrian = df_pedestrian.loc[df_pedestrian['NUMBER OF PEDESTRIANS KILLED']+df_pedestrian['NUMBER OF PEDESTRIANS INJURED']==0]
df_injury_pedestrian = df_pedestrian.loc[(df_pedestrian['NUMBER OF PEDESTRIANS INJURED']>0) & (df_pedestrian['NUMBER OF PEDESTRIANS KILLED'] == 0)]
df_killed_pedestrian = df_pedestrian.loc[df_pedestrian['NUMBER OF PEDESTRIANS KILLED']>0]
for i in range(0,len(focus_causes)):
    summen = 0 
    summen = sum(df_nonfatal_pedestrian['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_nonfatal_pedestrian['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_nonfatal_pedestrian['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_nonfatal_pedestrian['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_nonfatal_pedestrian['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    non_fatal_pedestrian[i] = summen
    summen = 0 
    summen = sum(df_injury_pedestrian['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_injury_pedestrian['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_injury_pedestrian['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_injury_pedestrian['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_injury_pedestrian['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    injury_pedestrian[i] = summen
    summen = 0 
    summen = sum(df_killed_pedestrian['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i])
    summen = summen + sum(df_killed_pedestrian['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i])
    summen = summen + sum(df_killed_pedestrian['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])
    summen = summen + sum(df_killed_pedestrian['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])
    summen = summen + sum(df_killed_pedestrian['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])
    killed_pedestrian[i] = summen


# In[143]:


df_distributions = pd.DataFrame()
df_distributions['Name'] = focus_causes
df_distributions['No injuries'] = non_fatal_pedestrian.astype(int)
df_distributions['With injuries'] = injury_pedestrian.astype(int)
df_distributions['Fatal'] = killed_pedestrian.astype(int)
df_distributions

df_total = df_distributions["No injuries"] + df_distributions["With injuries"] + df_distributions["Fatal"]
df_rel = df_distributions[df_distributions.columns[1:]].div(df_total, 0)

years = ["No injuries", "With injuries", "Fatal"]
colors = list(Category20[20][0:3])

data1 = {'Causes' : focus_causes,
        'No injuries'   : list(df_rel['No injuries']),
        'With injuries'   : list(df_rel['With injuries']),
        'Fatal'   : list(df_rel['Fatal'])}

p = figure(x_range=focus_causes, plot_width=1200,plot_height=1000, title="Causes and distributions of severity for accidents involving pedestrians",
           toolbar_location=None, tools="")

p.vbar_stack(years, x='Causes', width=0.9, color=colors, source=data1,
             legend_label=years)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "center_left"
p.legend.orientation = "vertical"
p.xaxis.major_label_orientation = "vertical"
p.yaxis.formatter = NumeralTickFormatter(format='0 %')
p.legend.label_text_font_size = "10px"
output_file("distibution_causes_pedestrians.html")
save(p)
#show(p)


# In[144]:


non_fatal_pedestrian = np.zeros(len(focus_vehicles))
injury_pedestrian = np.zeros(len(focus_vehicles))
killed_pedestrian = np.zeros(len(focus_vehicles))
df_nonfatal_pedestrian = df_pedestrian.loc[df_pedestrian['NUMBER OF PEDESTRIANS KILLED']+df_pedestrian['NUMBER OF PEDESTRIANS INJURED']==0]
df_injury_pedestrian = df_pedestrian.loc[(df_pedestrian['NUMBER OF PEDESTRIANS INJURED']>0) & (df_pedestrian['NUMBER OF PEDESTRIANS KILLED'] == 0)]
df_killed_pedestrian = df_pedestrian.loc[df_pedestrian['NUMBER OF PEDESTRIANS KILLED']>0]
for i in range(0,len(focus_vehicles)):
    summen = 0 
    summen = sum(df_nonfatal_pedestrian['VEHICLE TYPE CODE 1'] == focus_vehicles[i])
    summen = summen + sum(df_nonfatal_pedestrian['VEHICLE TYPE CODE 2'] == focus_vehicles[i])
    summen = summen + sum(df_nonfatal_pedestrian['VEHICLE TYPE CODE 3'] == focus_vehicles[i])
    summen = summen + sum(df_nonfatal_pedestrian['VEHICLE TYPE CODE 4'] == focus_vehicles[i])
    summen = summen + sum(df_nonfatal_pedestrian['VEHICLE TYPE CODE 5'] == focus_vehicles[i])
    non_fatal_pedestrian[i] = summen
    summen = 0 
    summen = sum(df_injury_pedestrian['VEHICLE TYPE CODE 1'] == focus_vehicles[i])
    summen = summen + sum(df_injury_pedestrian['VEHICLE TYPE CODE 2'] == focus_vehicles[i])
    summen = summen + sum(df_injury_pedestrian['VEHICLE TYPE CODE 3'] == focus_vehicles[i])
    summen = summen + sum(df_injury_pedestrian['VEHICLE TYPE CODE 4'] == focus_vehicles[i])
    summen = summen + sum(df_injury_pedestrian['VEHICLE TYPE CODE 5'] == focus_vehicles[i])
    injury_pedestrian[i] = summen
    summen = 0 
    summen = sum(df_killed_pedestrian['VEHICLE TYPE CODE 1'] == focus_vehicles[i])
    summen = summen + sum(df_killed_pedestrian['VEHICLE TYPE CODE 2'] == focus_vehicles[i])
    summen = summen + sum(df_killed_pedestrian['VEHICLE TYPE CODE 3'] == focus_vehicles[i])
    summen = summen + sum(df_killed_pedestrian['VEHICLE TYPE CODE 4'] == focus_vehicles[i])
    summen = summen + sum(df_killed_pedestrian['VEHICLE TYPE CODE 5'] == focus_vehicles[i])
    killed_pedestrian[i] = summen


# In[145]:


df_distributions = pd.DataFrame()
df_distributions['Name'] = focus_vehicles
df_distributions['No injuries'] = non_fatal_pedestrian.astype(int)
df_distributions['With injuries'] = injury_pedestrian.astype(int)
df_distributions['Fatal'] = killed_pedestrian.astype(int)
df_distributions

df_total = df_distributions["No injuries"] + df_distributions["With injuries"] + df_distributions["Fatal"]
df_rel = df_distributions[df_distributions.columns[1:]].div(df_total, 0)

years = ["No injuries", "With injuries", "Fatal"]
colors = list(Category20[19][0:3])

data1 = {'Causes' : focus_vehicles,
        'No injuries'   : list(df_rel['No injuries']),
        'With injuries'   : list(df_rel['With injuries']),
        'Fatal'   : list(df_rel['Fatal'])}

p = figure(x_range=focus_vehicles, plot_width=1200,plot_height=1000, title="Vehicle types and distributions of severity for accidents involving pedestrians",
           toolbar_location=None, tools="")

p.vbar_stack(years, x='Causes', width=0.9, color=colors, source=data1,
             legend_label=years)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "center_left"
p.legend.orientation = "vertical"
p.xaxis.major_label_orientation = "vertical"
p.yaxis.formatter = NumeralTickFormatter(format='0 %')
p.legend.label_text_font_size = "10px"
output_file("distibution_vehicles_pedestrians.html")
save(p)


# In[146]:


df_relative = pd.DataFrame()
df_relative['Name'] = focus_causes
df_relative['Pedestrian'] = np.zeros(len(focus_causes))
df_relative['Cyclist'] = np.zeros(len(focus_causes))

for i in range(0,len(focus_causes)):
    if focus_causes[i] != 'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':
        df_relative['Cyclist'].iloc[i] = (sum((df_cyclist['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_cyclist['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_cyclist['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_cyclist['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_cyclist['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i]))/len(df_cyclist))/((sum((filterdf['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (filterdf['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(filterdf['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(filterdf['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(filterdf['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i]))/len(filterdf)))
        df_relative['Pedestrian'].iloc[i] = (sum((df_pedestrian['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_pedestrian['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_pedestrian['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_pedestrian['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_pedestrian['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i]))/len(df_pedestrian))/((sum((filterdf['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (filterdf['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(filterdf['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(filterdf['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(filterdf['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i]))/len(filterdf)))
df_relative.set_index('Name',inplace=True)
df_relative.drop('Pedestrian/Bicyclist/Other Pedestrian Error/Confusion',inplace=True)


# In[147]:


df_relative.index.name = None
df_relative


# In[148]:


from matplotlib import cm
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(30,10))
fig.subplots_adjust(hspace=0.2)
color = np.flip(cm.plasma(np.linspace(.2,.6, 10)))
df_relative.plot(subplots=True, ax=axes, kind='bar',color=color)

_ = plt.plot()
fig.savefig('Causes_in_df_relative.png', dpi=800)
plt.show()


# # Frequency plots

# In[149]:


df_abs = pd.DataFrame()
df_abs['Name'] = focus_causes
df_abs['Pedestrian'] = np.zeros(len(focus_causes))
df_abs['Cyclist'] = np.zeros(len(focus_causes))
df_abs['All accidents'] = np.zeros(len(focus_causes))

for i in range(0,len(focus_causes)):
    #if focus_causes[i] != 'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':
    df_abs['Cyclist'].iloc[i] = (sum((df_cyclist['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_cyclist['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_cyclist['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_cyclist['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_cyclist['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))
    df_abs['Pedestrian'].iloc[i] = (sum((df_pedestrian['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_pedestrian['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_pedestrian['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_pedestrian['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_pedestrian['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))
    df_abs['All accidents'].iloc[i] = (sum((filterdf['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (filterdf['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(filterdf['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(filterdf['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(filterdf['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))
                                 
df_abs.set_index('Name',inplace=True)
df_abs.index.name = None
#df_relative.drop('Pedestrian/Bicyclist/Other Pedestrian Error/Confusion',inplace=True)


# In[150]:


fig, axes = plt.subplots(nrows=3, ncols=1,figsize=(30,30))
fig.subplots_adjust(hspace=0.2)
color = np.flip(cm.plasma(np.linspace(.2,.6, 10)))
df_abs.plot(subplots=True, ax=axes, kind='bar',color=color)

_ = plt.plot()
#fig.savefig('Causes_in_df_relative.png', dpi=800)
plt.show()


# In[233]:


df_rel = pd.DataFrame()
total_filter = filterdf.iloc[:,18:23].notna().astype('uint8').sum().sum()
total_pedestrian = df_pedestrian.iloc[:,18:23].notna().astype('uint8').sum().sum()
total_cyclist = df_cyclist.iloc[:,18:23].notna().astype('uint8').sum().sum()
df_rel['Name'] = focus_causes
df_rel['Pedestrian'] = np.zeros(len(focus_causes))
df_rel['Cyclist'] = np.zeros(len(focus_causes))
df_rel['All accidents'] = np.zeros(len(focus_causes))

for i in range(0,len(focus_causes)):
    #if focus_causes[i] != 'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':
    df_rel['Cyclist'].iloc[i] = (sum((df_cyclist['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_cyclist['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_cyclist['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_cyclist['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_cyclist['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/total_cyclist
    df_rel['Pedestrian'].iloc[i] = (sum((df_pedestrian['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (df_pedestrian['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(df_pedestrian['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(df_pedestrian['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(df_pedestrian['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/total_pedestrian
    df_rel['All accidents'].iloc[i] = (sum((filterdf['CONTRIBUTING FACTOR VEHICLE 1'] == focus_causes[i]) + (filterdf['CONTRIBUTING FACTOR VEHICLE 2'] == focus_causes[i]) +(filterdf['CONTRIBUTING FACTOR VEHICLE 3'] == focus_causes[i])+(filterdf['CONTRIBUTING FACTOR VEHICLE 4'] == focus_causes[i])+(filterdf['CONTRIBUTING FACTOR VEHICLE 5'] == focus_causes[i])))/total_filter
                                 
df_rel.set_index('Name',inplace=True)
df_rel.index.name = None
#df_relative.drop('Pedestrian/Bicyclist/Other Pedestrian Error/Confusion',inplace=True)


# In[152]:


fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(30,10))
fig.subplots_adjust(hspace=0.2)
color = np.flip(cm.plasma(np.linspace(.2,.6, 10)))
df_rel.plot(subplots=True, ax=axes, kind='bar',color=color)

_ = plt.plot()
#fig.savefig('Causes_in_df_relative.png', dpi=800)
plt.show()


# In[153]:


#df_day_pedestrian
df_rel.index.rename('Time',inplace=True)
df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
df11 = df_rel.copy()

hours = focus_causes1#[str(round(x)) for x in np.linspace(0, len(focus_causes)-1, num = len(focus_causes))]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(df_rel.columns):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category10[3][indx])

p.title.text='Contributing factors frequency in data sets'
#p.xaxis.axis_label='Hour of the day'
p.yaxis.axis_label='Frequency of contributing factors for data sets'
p.legend.orientation = "vertical"
p.xaxis.major_label_orientation = np.pi/3
#p.xaxis.major_label_orientation = "vertical"
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
p.yaxis.formatter = NumeralTickFormatter(format='0 %')
#p.legend.orientation = "horizontal"
p.legend.label_text_font_size = "10px"

output_file("causes_frequency_datasets.html")
save(p)
#show(p)


# In[154]:


df_pedestrian.iloc[:,-6:-1]


# In[155]:


df_rel = pd.DataFrame()
total_filter = filterdf.iloc[:,-6:-1].notna().astype('uint8').sum().sum()
total_pedestrian = df_pedestrian.iloc[:,-6:-1].notna().astype('uint8').sum().sum()
total_cyclist = df_cyclist.iloc[:,-6:-1].notna().astype('uint8').sum().sum()
df_rel['Name'] = focus_vehicles
df_rel['Pedestrian'] = np.zeros(len(focus_vehicles))
df_rel['Cyclist'] = np.zeros(len(focus_vehicles))
df_rel['All accidents'] = np.zeros(len(focus_vehicles))

for i in range(0,len(focus_vehicles)):
    #if focus_causes[i] != 'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':
    df_rel['Cyclist'].iloc[i] = (sum((df_cyclist['VEHICLE TYPE CODE 1'] == focus_vehicles[i]) + (df_cyclist['VEHICLE TYPE CODE 2'] == focus_vehicles[i]) +(df_cyclist['VEHICLE TYPE CODE 3'] == focus_vehicles[i])+(df_cyclist['VEHICLE TYPE CODE 4'] == focus_vehicles[i])+(df_cyclist['VEHICLE TYPE CODE 5'] == focus_vehicles[i])))/total_cyclist
    df_rel['Pedestrian'].iloc[i] = (sum((df_pedestrian['VEHICLE TYPE CODE 1'] == focus_vehicles[i]) + (df_pedestrian['VEHICLE TYPE CODE 2'] == focus_vehicles[i]) +(df_pedestrian['VEHICLE TYPE CODE 3'] == focus_vehicles[i])+(df_pedestrian['VEHICLE TYPE CODE 4'] == focus_vehicles[i])+(df_pedestrian['VEHICLE TYPE CODE 5'] == focus_vehicles[i])))/total_pedestrian
    df_rel['All accidents'].iloc[i] = (sum((filterdf['VEHICLE TYPE CODE 1'] == focus_vehicles[i]) + (filterdf['VEHICLE TYPE CODE 2'] == focus_vehicles[i]) +(filterdf['VEHICLE TYPE CODE 3'] == focus_vehicles[i])+(filterdf['VEHICLE TYPE CODE 4'] == focus_vehicles[i])+(filterdf['VEHICLE TYPE CODE 5'] == focus_vehicles[i])))/total_filter
                                 
df_rel.set_index('Name',inplace=True)
df_rel.index.name = None


# In[156]:


df_rel.index.rename('Time',inplace=True)
#df11.rename(columns={'Pedestrian/Bicyclist/Other Pedestrian Error/Confusion':'Pedestrian/Bicyclist Error/Confusion'},inplace=True)
df11 = df_rel.copy()

hours = focus_vehicles#[str(round(x)) for x in np.linspace(0, len(focus_causes)-1, num = len(focus_causes))]
source = ColumnDataSource(df11)
p = figure(x_range = FactorRange(factors=hours),plot_width=900)
bar ={} # to store vbars
### here we will do a for loop:
for indx,i in enumerate(df_rel.columns):
    bar[i] = p.vbar(x='Time',  top=i, source= source, legend_label=i, width=0.8, alpha=1.2, muted_alpha=0.04, muted = True,color=Category10[3][indx])

p.title.text='Involved vehicle types frequency in data sets'
#p.xaxis.axis_label='Hour of the day'
p.yaxis.axis_label='Frequency of involved vehicle types for data sets'
p.legend.orientation = "vertical"
p.xaxis.major_label_orientation = np.pi/3
#p.xaxis.major_label_orientation = "vertical"
p.add_layout(p.legend[0], 'right')
p.legend.click_policy="mute"
p.yaxis.formatter = NumeralTickFormatter(format='0 %')
#p.legend.orientation = "horizontal"
p.legend.label_text_font_size = "10px"

output_file("vehicles_frequency_datasets.html")
save(p)
#show(p)


# # Mapping worst spots

# In[157]:


#Counter(df_cyclist.loc[df_cyclist['CONTRIBUTING FACTOR VEHICLE 1'] == 'Failure to Yield Right-of-Way']['LOCATION']).most_common()
df_both = pd.concat([df_cyclist,df_pedestrian])
tri = Counter(df_both.loc[(df_both['CONTRIBUTING FACTOR VEHICLE 1'] == 'Failure to Yield Right-of-Way') | (df_both['CONTRIBUTING FACTOR VEHICLE 2'] == 'Failure to Yield Right-of-Way') |(df_both['CONTRIBUTING FACTOR VEHICLE 3'] == 'Failure to Yield Right-of-Way') |(df_both['CONTRIBUTING FACTOR VEHICLE 4'] == 'Failure to Yield Right-of-Way') | (df_both['CONTRIBUTING FACTOR VEHICLE 5'] == 'Failure to Yield Right-of-Way')]['LOCATION']).most_common()[2:16]
tri3 = Counter(df_both.loc[(df_both['CONTRIBUTING FACTOR VEHICLE 1'] == 'Driver Inattention/Distraction') | (df_both['CONTRIBUTING FACTOR VEHICLE 2'] == 'Driver Inattention/Distraction') |(df_both['CONTRIBUTING FACTOR VEHICLE 3'] == 'Driver Inattention/Distraction') |(df_both['CONTRIBUTING FACTOR VEHICLE 4'] == 'Driver Inattention/Distraction') | (df_both['CONTRIBUTING FACTOR VEHICLE 5'] == 'Driver Inattention/Distraction')]['LOCATION']).most_common()[2:32]
tri2 = Counter(df_both.loc[(df_both['CONTRIBUTING FACTOR VEHICLE 1'] == 'Failure to Yield Right-of-Way') | (df_both['CONTRIBUTING FACTOR VEHICLE 2'] == 'Failure to Yield Right-of-Way') |(df_both['CONTRIBUTING FACTOR VEHICLE 3'] == 'Failure to Yield Right-of-Way') |(df_both['CONTRIBUTING FACTOR VEHICLE 4'] == 'Failure to Yield Right-of-Way') | (df_both['CONTRIBUTING FACTOR VEHICLE 5'] == 'Failure to Yield Right-of-Way')]['LOCATION']).most_common()[2:]
len(tri2[9000:])
len(tri2[2:])
#tri2 = Counter(df_both.loc[(df_both['CONTRIBUTING FACTOR VEHICLE 1'] == 'Failure to Yield Right-of-Way') | (df_both['CONTRIBUTING FACTOR VEHICLE 2'] == 'Failure to Yield Right-of-Way') |(df_both['CONTRIBUTING FACTOR VEHICLE 3'] == 'Failure to Yield Right-of-Way') |(df_both['CONTRIBUTING FACTOR VEHICLE 4'] == 'Failure to Yield Right-of-Way') | (df_both['CONTRIBUTING FACTOR VEHICLE 5'] == 'Failure to Yield Right-of-Way')]['LOCATION']).most_common()[2:16]
#Counter(df_both.loc[(df_both['CONTRIBUTING FACTOR VEHICLE 1'] == 'Failure to Yield Right-of-Way') | (df_both['CONTRIBUTING FACTOR VEHICLE 2'] == 'Failure to Yield Right-of-Way') |(df_both['CONTRIBUTING FACTOR VEHICLE 3'] == 'Failure to Yield Right-of-Way') |(df_both['CONTRIBUTING FACTOR VEHICLE 4'] == 'Failure to Yield Right-of-Way') | (df_both['CONTRIBUTING FACTOR VEHICLE 5'] == 'Failure to Yield Right-of-Way')]['LONGITUDE']).most_common()[2:35]
#len(latlon2)


# In[158]:


latlon = pd.DataFrame(0,index=np.arange(len(tri)),columns=['Lat','Lon','Accidents','Injuries','Fatalities'])
for i in range(0,len(tri)):
    latlon['Lat'].iloc[i] = float(tri[i][0].split("(")[1].split(",")[0])
    latlon['Lon'].iloc[i] = float(tri[i][0].split("(")[1].split(",")[1].split(")")[0])
    latlon['Accidents'].iloc[i] = tri[i][1]
    latlon['Injuries'].iloc[i] = df_both.loc[df_both['LOCATION'] == tri[i][0]][['NUMBER OF CYCLIST INJURED','NUMBER OF PEDESTRIANS INJURED']].sum().sum()
    latlon['Fatalities'].iloc[i] = df_both.loc[df_both['LOCATION'] == tri[i][0]][['NUMBER OF CYCLIST KILLED','NUMBER OF PEDESTRIANS KILLED']].sum().sum()

latlon2 = pd.DataFrame(0,index=np.arange(len(tri3)),columns=['Lat','Lon','Accidents','Injuries','Fatalities'])
for i in range(0,len(tri3)):
    latlon2['Lat'].iloc[i] = float(tri3[i][0].split("(")[1].split(",")[0])
    latlon2['Lon'].iloc[i] = float(tri3[i][0].split("(")[1].split(",")[1].split(")")[0])
    latlon2['Accidents'].iloc[i] = tri3[i][1]
    latlon2['Injuries'].iloc[i] = df_both.loc[df_both['LOCATION'] == tri3[i][0]][['NUMBER OF CYCLIST INJURED','NUMBER OF PEDESTRIANS INJURED']].sum().sum()
    latlon2['Fatalities'].iloc[i] = df_both.loc[df_both['LOCATION'] == tri3[i][0]][['NUMBER OF CYCLIST KILLED','NUMBER OF PEDESTRIANS KILLED']].sum().sum()


# latlon2 = pd.DataFrame(0,index=np.arange(len(tri2)),columns=['Lat','Lon','Accidents','Injuries','Fatalities'])
# for i in range(0,len(tri2)):
#     latlon2['Lat'].iloc[i] = float(tri2[i][0].split("(")[1].split(",")[0])
#     latlon2['Lon'].iloc[i] = float(tri2[i][0].split("(")[1].split(",")[1].split(")")[0])
#     latlon2['Accidents'].iloc[i] = tri2[i][1]
#     latlon2['Injuries'].iloc[i] = df_both.loc[df_both['LOCATION'] == tri2[i][0]][['NUMBER OF CYCLIST INJURED','NUMBER OF PEDESTRIANS INJURED']].sum().sum()
#     latlon2['Fatalities'].iloc[i] = df_both.loc[df_both['LOCATION'] == tri2[i][0]][['NUMBER OF CYCLIST KILLED','NUMBER OF PEDESTRIANS KILLED']].sum().sum()

# In[159]:


map_hooray = folium.Map(location=[40.756470, -73.953557],
                    zoom_start = 11, parse_html=True) # Uses lat then lon. The bigger the zoom number, the closer in you get
#folium.Marker([latlon.Lat, latlon.Lon], popup='SF city hall').add_to(map_hooray)
for i in range(0, len(latlon)):
    folium.Marker([latlon['Lat'].iloc[i],latlon['Lon'].iloc[i]],popup=(
    "Accidents: {accidents} <br>"
    "Injuries: {injury} <br>"
    "Fatalities: {fatal} <br>"
    ).format(accidents=latlon['Accidents'].iloc[i],injury=latlon['Injuries'].iloc[i],fatal=latlon['Fatalities'].iloc[i]), parse_html=True).add_to(map_hooray)
map_hooray # Calls the map to display # 
#map_hooray.save('worst_yield_right.html')


# In[160]:


map_hooray = folium.Map(location=[40.729974, -73.960431],
                    zoom_start = 11, parse_html=True) # Uses lat then lon. The bigger the zoom number, the closer in you get
#folium.Marker([latlon.Lat, latlon.Lon], popup='SF city hall').add_to(map_hooray)
for i in range(0, len(latlon2)):
    folium.Marker([latlon2['Lat'].iloc[i],latlon2['Lon'].iloc[i]],popup=(
    "Accidents: {accidents} <br>"
    "Injuries: {injury} <br>"
    "Fatalities: {fatal} <br>"
    ).format(accidents=latlon2['Accidents'].iloc[i],injury=latlon2['Injuries'].iloc[i],fatal=latlon2['Fatalities'].iloc[i]), parse_html=True).add_to(map_hooray)
map_hooray # Calls the map to display # 
#map_hooray.save('worst_driver_distraction.html')

