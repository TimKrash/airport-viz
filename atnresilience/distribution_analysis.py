import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Script to plot distribution of data. Plots box-plots and histograms"""
file = 2012  # Change this to the relevant year
script_dir = os.path.dirname(os.getcwd())
### Uncomment next line for specific files, CC for cancelled, ADM for Arrival delay means etc.
rel_path = "data\\processed\\%s_CC_airports.csv" % file
# rel_path = "data\\processed\\%s_combined.csv" % file  # This is for all of the years combined
raw_file = os.path.join(script_dir, rel_path)
### Filters original file to only keep required fields to save memory and processing time
fields = ["Destination_Airport_Code", "Flight_Date", "Date_Time", "Day_Of_Year", "Origin_Airport_Code", "Arr_Delay",
          "Can_Status", "Div_Status", "Dep_Delay","Combined_Arr_Delay"]
df = pd.read_csv(raw_file,usecols=fields)
### Histograms for delays
"""Filtering to the busiest airports in the first line with the count in the square brackets and converts it to a list.
Another option is to provide the list of airports directly to 'hi_volume_names'. Switch between arrival and departure
delays by replacing 'Origin' with 'Destination' wherever variable appears. Also make sure to change relevant variables 
such as 'Dep_Delay' or 'Arr_Delay'. Same applies for box-plots.
"""
# hi_volume = df['Origin_Airport_Code'].value_counts()[:40]
# hi_volume_names = hi_volume.index.tolist()
hi_volume_names = ['ATL', 'ORD', 'DEN']
for k in hi_volume_names:
    print(k)
    hi_volume_airports = df[df['Origin_Airport_Code']==k]
    hi_volume_airports_pivots = hi_volume_airports.pivot_table(index='Flight_Date', columns='Origin_Airport_Code',
                                                               values='Dep_Delay')
    airport_bins = np.arange(-20,120,5) # Bin size is last value with min and max range as first two values respectively
    hi_volume_airports_pivots.plot(kind='hist', bins=airport_bins, alpha=.4, legend=True)
    plt.show()

### Box plots for delays
# hi_volume = df['Origin_Airport_Code'].value_counts()[:50]
# hi_volume_names = hi_volume.index.tolist()
# hi_volume_names = ['ATL', 'ORD', 'LAX']
# df = df[df['Destination_Airport_Code'].isin(hi_volume_names)]  # Filters to keep only a/ps from list both directions
# df = df[df['Origin_Airport_Code'].isin(hi_volume_names)]
# df = df[df['Destination_Airport_Code'].isin(hi_volume_names)]
# hi_volume_airports = df.groupby(['Flight_Date', 'Origin_Airport_Code'], as_index=False)[
#                 ['Combined_Arr_Delay']].mean()
# hi_volume_airports_pivots = hi_volume_airports.pivot_table(index='Flight_Date', columns='Origin_Airport_Code',
#                                                            values='Combined_Arr_Delay')
# hi_volume_airports_pivots.plot(kind='box')
# plt.show()

### Histogram for cancellations
# hi_volume_names = ['ATL', 'ORD', 'DEN']
# for k in hi_volume_names:
#     print(k)
#     hi_volume_airports = df[df['Destination_Airport_Code']==k]
#     hi_volume_airports_group = hi_volume_airports.groupby(['Flight_Date', 'Destination_Airport_Code'], as_index=False)
#     [['Can_Status']].sum()
#     hi_volume_airports_pivots = hi_volume_airports_group.pivot_table(index='Flight_Date',
#                                                                      columns='Destination_Airport_Code',
#                                                                      values='Can_Status')
#     airport_bins = np.arange(-20,300,3)
#     hi_volume_airports_pivots.plot(kind='hist', bins=airport_bins, alpha=.4, legend=True)
#     plt.show()

### Box plots for cancellations
# hi_volume_names = ['ATL', 'ORD', 'LAX']
# df = df[df['Destination_Airport_Code'].isin(hi_volume_names)]
# df = df[df['Origin_Airport_Code'].isin(hi_volume_names)]
    # hi_volume_airports_group = df.groupby(['Flight_Date', 'Destination_Airport_Code'], as_index=False)[
#                 ['Can_Status']].sum()
# print(hi_volume_airports_group)
# hi_volume_airports_pivots = hi_volume_airports_group.pivot_table(index='Flight_Date',
#                                                                  columns='Destination_Airport_Code',
#                                                                  values='Can_Status')
# print(hi_volume_airports_pivots.describe())
# hi_volume_airports_pivots.plot(kind='box')
# plt.show()