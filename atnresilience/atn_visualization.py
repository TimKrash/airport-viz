import os
import math
import datetime
import calendar

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np
import networkx as nx
import sqlite3
from sqlalchemy import *

import atn_tools as oc
import atn_analysis as analysis
import db_tools
# from atn_analysis import AnalysisQueries

from mpl_toolkits.basemap import Basemap
# Ignore matplotlib warnings from Basemap. Must use matplotlib<3.1.3
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# The default Times New Roman pack is bold, the following fixes the default to regular font
# The two commands only need to be run once per local instance
# https://github.com/matplotlib/matplotlib/issues/5574
# del matplotlib.font_manager.weight_dict['roman']
# matplotlib.font_manager._rebuild()

# Set global plot formatting
matplotlib.style.use('classic')
plt.rcParams["font.family"] = "Times New Roman" #Set font type for all plots
sns.set_palette("colorblind") #set color scheme for all plots
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8)
matplotlib.rc('legend',labelspacing=0.15,fontsize=10,edgecolor='0.8')
# matplotlib.rc('figure', facecolor='White')
# matplotlib.rc('axes', facecolor='White')
# matplotlib.rc('savefig', facecolor='White')

def airport_stats(years, airline_list, dropped = True):
    stats_df = pd.DataFrame(columns=['Destination_Airport_Code','FlightCount','ValidDays'])

    for airline in airline_list:
        for year in years:
            if dropped:
                airline_sql = '''
                    SELECT Destination_Airport_Code,
                    COUNT(*)/:days_in_year AS FlightCount,
                    COUNT(DISTINCT(Day_Of_Year)) AS ValidDays
                    FROM atn_performance
                    WHERE Year = :year
                    AND Unique_Carrier_ID LIKE :airline
                    AND Can_Status != 1 
                    AND Div_Status != 1 
                    GROUP BY Destination_Airport_Code
                    HAVING COUNT(DISTINCT(Day_Of_Year)) < :days_in_year
                    '''
            else:
                airline_sql = '''
                    SELECT Destination_Airport_Code,
                    COUNT(*)/:days_in_year AS FlightCount,
                    COUNT(DISTINCT(Day_Of_Year)) AS ValidDays
                    FROM atn_performance
                    WHERE Year = :year
                    AND Unique_Carrier_ID LIKE :airline
                    AND Can_Status != 1 
                    AND Div_Status != 1 
                    GROUP BY Destination_Airport_Code
                    HAVING COUNT(DISTINCT(Day_Of_Year)) = :days_in_year
                    '''
            if calendar.isleap(year) == 1:
                no_of_days = 366
            else:
                no_of_days = 365

            params={'year' : year, 'days_in_year' : no_of_days, 'airline' : airline}
            stats_df = stats_df.append(db_tools.DBQueries().query_to_df(airline_sql, params=params), ignore_index=True)
            
    stats_df = stats_df.astype({'FlightCount':'int64','ValidDays':'int64'}, inplace=True)
    plt.hist(stats_df['FlightCount'], color='k')
    plt.title('Average Flight Count')
    plt.ylim((0,250))
        
def rolling_mean_plot(
        years,
        airline_list,
        include_data,
        processed_direc,
        stat_type,
        window = 31,
        y_upper = 500):
    """
    Creates a rolling mean plot of Mahalanobis distance for the years specified. 
    
    Must generate MD data first from the atn_analysis

    Parameters
    ----------
    years: list
        Years to plot
    window: int
        The window for the rolling mean
    airline_list: list
        Airlines to plot
    include_data: string
        Include data type
    processed_direc: string
        Path to the processed file folder
    stat_type: The type of statistic being plotted
        Options include:
            md - Mahalanobis distance
            airline_z - Z-score sorted by airline network
            delay - total daily delay
    
    Returns
    -------
    A matplotlib figure
    
    Notes
    -----
    
    """
    
    maha_dict = {"ADD" : "Arrival_Div_Delay", "ADM":"Arrival_Delay", "DCC":"Delay_Cancel_Count","CC":"Cancel Count", "DD":"Departure_Delay"} 

    stat_params = {
        'md': {
            'filename_pattern'  : 'MDdata_%s.csv'%(include_data),
            'y_label'           : 'Mahalanobis distance',
            'agg_col'           : maha_dict[include_data]
            },
        'airline_z' : {
            'filename_pattern'  : 'airline_Zdata_%s.csv'%(include_data),
            'y_label'           : 'Z-Score',
            'agg_col'           : 'Zscore'
            },
        'delay'     : {
            'filename_pattern'  : 'daily_delay.csv',
            'y_label'           : 'Mean',
            'agg_col'           : 'Pos_Arr_Delay'
            },
        }

    #Used to define include_data parameters to the column names
    poi_column = stat_params[stat_type]['agg_col']
    all_plot = False
    
    df_mean_airline = pd.DataFrame(columns=['Date_Time','Airline',poi_column]) #Initialize a df to take the mean for all years
    #Define figure size
    fig, ax = plt.subplots(figsize=(5.5,2.5))
    
    if 'ALL' in airline_list:
        df_mean_all = pd.DataFrame(columns=['Date_Time','Airline',poi_column]) #Initialize a df to take the mean for all years
        
        for year in years:  
            file_path = '%s%s_%s_%s'%(processed_direc,'ALL',year,stat_params[stat_type]['filename_pattern'])
            md_df = pd.read_csv(file_path)
            #Add a date columne that will use stndard YYYY-MM-DD format
            md_df['Date_Time'] = pd.to_datetime(year * 1000 + md_df['Day_of_Year'], format='%Y%j')
            md_df['Airline'] = 'ALL'
            
            #Drop the now unneeded Day_of_Year column
            md_df = md_df.drop(columns=['Day_of_Year'])
            df_mean_all = pd.concat([df_mean_all,md_df],ignore_index=True, sort=False)
            
        df_mean_all['Mean'] = df_mean_all[poi_column].rolling(window,center=True).mean()    
        airline_list.remove('ALL')
        all_plot = True
            
    for airline in airline_list:
        for year in years:  
            file_path = '%s%s_%s_%s'%(processed_direc,airline,year,stat_params[stat_type]['filename_pattern'])
            md_df = pd.read_csv(file_path)
            #Add a date columne that will use stndard YYYY-MM-DD format
            md_df['Date_Time'] = pd.to_datetime(year * 1000 + md_df['Day_of_Year'], format='%Y%j')
            md_df['Airline'] = airline
            
            #Drop the now unneeded Day_of_Year column
            md_df = md_df.drop(columns=['Day_of_Year'])
            
            df_mean_airline = pd.concat([df_mean_airline,md_df],ignore_index=True, sort=False)
            
        df_mean_airline['Mean'] = df_mean_airline[poi_column].rolling(window,center=True).mean()
           
    sns.lineplot(ax=ax, x='Date_Time',y='Mean',hue='Airline',style='Airline',data=df_mean_airline)   
    if all_plot:
        ax2 = plt.twinx()
        #alpha sets transparency. 1 is full, 0 is invisible
        sns.lineplot(ax=ax2, x='Date_Time',y='Mean',color='gray',alpha=0.5,data=df_mean_all,label='ALL',linewidth=8.0) 
        # ax2.set_ylim(top=20)
        ax2.set_ylabel(stat_params[stat_type]['y_label'])
        ax2.yaxis.set_tick_params(labelsize=8)
        ax2.legend(loc = 'upper right')

    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel(stat_params[stat_type]['y_label'],fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4,7,10]))
    ax.set_xlim([datetime.date(years[0], 1, 1), datetime.date(years[-1], 11, 1)])
    ax.set_ylim(top=y_upper)
    # Remove legend titles
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc = 'upper left', handles=handles[1:], labels=labels[1:])
    fig.tight_layout() #reduce margin white space
    
def md_median_lineplot(years, airline_list, include_data, processed_direc):
    """
    Creates a median plot of Mahalanobis distance for the years specified with upper and lower quantiles. 
    Must generate MD data first from the atn_analysis

    Parameters
    ----------
    years: list
        Years to plot
    airline_list: list
        Airlines to plot
    include_data: string
        Include data type
    processed_direc: string
        Path to the processed file folder
    
    Returns
    -------
    A matplotlib figure
    
    Notes
    -----
    
    """

    maha_dict = {"ADD" : "Arrival_Div_Delay", "ADM":"Arrival_Delay", "DCC":"Delay_Cancel_Count","CC":"Cancel Count", "DD":"Departure_Delay"} #Used to define include_data parameters to the column names
    poi_column = maha_dict[include_data]
    
    graph_df = pd.DataFrame(columns = ['Airline','Year','Year_Quarter',poi_column])
    #Define figure size
    fig, ax = plt.subplots(figsize=(5.5,2.5))
    # Argument used to determine if all will be plotted
    all_plot = 0
    
    # Used to map quarters to months for plotting
    quarter_mapping = {1:1, 2:4, 3:7, 4:10}
    
    if 'ALL' in airline_list:
        all_plot = 1  #Change all plot argument so entire network data will be plotted
        airline_list.remove('ALL') #Remove all from the airline list so it is not passed in loop for airlines
        df_mean_all = pd.DataFrame(columns = ['Airline','Year','Year_Quarter',poi_column])
        
        for year in years:
            graph_direc = processed_direc + '%s_%s_MDdata_%s.csv' %('ALL',year,include_data)
            df_import = pd.read_csv(graph_direc)
            #Add a datetime column
            df_import['Date_Time'] = pd.to_datetime(year * 1000 + df_import['Day_of_Year'], format='%Y%j')
            #Determine the quarter
            df_import['Quarter'] = df_import['Date_Time'].dt.quarter
            
#            return(df_import) ##
            
            for quarter in range(1,5):
                #Calculate the Quartile Data
                quantile_data = df_import.loc[df_import['Quarter'] == quarter]
                quantile_data = quantile_data.drop(columns=['Day_of_Year'])
                quantile_calc = quantile_data.quantile([0.5])
                # Add the corresponding year, quarter, and year_quarter(a single variable which denotes both the year and quarter)
                quantile_calc['Year'] = year
                quantile_calc['Quarter'] = quarter
                # Turn the year+quarter into datetime format                
                quantile_calc['Year_Quarter'] = pd.to_datetime(year * 100 + quarter_mapping[quarter], format='%Y%m')

                df_mean_all = pd.concat([df_mean_all,quantile_calc],ignore_index=True,sort=False)
                
    for airline in airline_list:
        #Create a df which will take the median and quantile data for each year
        airline_df = pd.DataFrame(columns = ['Year','Quarter',poi_column])
        
        for year in years:
            graph_direc = processed_direc + '%s_%s_MDdata_%s.csv' %(airline,year,include_data)
            df_import = pd.read_csv(graph_direc)
            # Add a datetime column
            df_import['Date_Time'] = pd.to_datetime(year * 1000 + df_import['Day_of_Year'], format='%Y%j')
            # Determine the quarter
            df_import['Quarter'] = df_import['Date_Time'].dt.quarter
            
            
            for quarter in range(1,5):
                #Calculate the Quartile Data
                quantile_data = df_import.loc[df_import['Quarter'] == quarter]
                quantile_data = quantile_data.drop(columns=['Day_of_Year'])
                quantile_calc = quantile_data.quantile([0.25,0.5,0.75])
                #Add the corresponding year, quarter, and year_quarter(a single variable which denotes both the year and quarter)
                quantile_calc['Year'] = year
                quantile_calc['Quarter'] = quarter
                #Turn the year+quarter into datetime format
                quantile_calc['Year_Quarter'] = pd.to_datetime(year * 100 + quarter_mapping[quarter], format='%Y%m')


                airline_df = pd.concat([airline_df,quantile_calc],ignore_index=True,sort=False)
        airline_df['Airline'] = airline
        graph_df = pd.concat([graph_df,airline_df],ignore_index=True,sort=False)
        
    sns.lineplot(ax=ax, x='Year_Quarter', y=poi_column, hue='Airline', style='Airline', data = graph_df)
    if all_plot == 1:
        ax2 = plt.twinx()
        sns.lineplot(ax=ax2, x='Year_Quarter',y=poi_column,color='gray',alpha=0.3,data=df_mean_all,label='ALL',linewidth=10.0) 
        ax2.set_ylabel('Mahalabonis distance')
        ax2.set_ylim(top=20)
        ax2.yaxis.set_tick_params(labelsize=8)

    
    ax.set_ylabel('Mahalabonis distance',fontsize=10)
    ax.set_xlabel('',fontsize=10)
    ax.set_xlim([datetime.date(2015, 1, 1), datetime.date(2017, 11, 1)])

    ax.legend(loc = 'upper left',fontsize=10)
    #Set the spacing of the major and minor x-axis ticks.
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4,7,10]))
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc = 'upper left', labelspacing=0.15,
        fontsize=10, handles=handles[1:], labels=labels[1:])
    fig.tight_layout() #reduce margin white space
    
def delay_mean_plot(years, airline_list, processed_direc):
    """
    Creates a plot of the total daily mean delay for the years specified. 
    Must generate total delay data from atn_tools first

    Parameters
    ----------
    years: list
        Years to plot
    airline_list: list
    processed_direc: string
        Path to the processed file folder
    
    Returns
    -------
    A matplotlib figure
    
    Notes
    -----
    
    """
    fig, ax = plt.subplots(figsize=(5.5,2.5))
    all_df = pd.DataFrame(columns=['Date','Airline','Pos_Arr_Delay'])
    
    all_plot = 0
    
    if 'ALL' in airline_list:
        airline_list.remove('ALL')
        df_mean_all = pd.DataFrame(columns=['Date','Airline','Pos_Arr_Delay']) #Initialize a df to take the mean for all years
        
        for year in years:
            all_data_direc = processed_direc + '%s__daily_delay.csv' %(year)
            df_import_all = pd.read_csv(all_data_direc)
            
            #Add a date columne that will use stndard YYYY-MM-DD format
            df_import_all['Date'] = pd.to_datetime(year*1000+df_import_all['Day'], format='%Y%j')
            df_import_all['Airline'] = 'ALL'
            #Drop the now unneeded Day column
            df_import_all.drop(columns='Day')
            
            df_mean_all = pd.concat([df_mean_all,df_import_all], ignore_index=True,sort=False)
#        return(df_mean_all)
        all_plot = 1
    
    for airline in airline_list:
        for year in years:
            data_direc = processed_direc + '%s_%s_daily_delay.csv' %(year,airline)
            df_import = pd.read_csv(data_direc)
            df_import['Date'] = pd.to_datetime(year*1000+df_import['Day'], format='%Y%j')
            df_import['Airline'] = airline
            df_import.drop(columns='Day')
            all_df = pd.concat([all_df,df_import], ignore_index=True,sort=False)
#            return(df_import)
    sns.lineplot(ax=ax, x='Date', y='Pos_Arr_Delay', hue='Airline', style='Airline', data = all_df)
    
    if all_plot == 1:
        ax2 = plt.twinx()
        #alpha sets transparency. 1 is full, 0 is invisible
        sns.lineplot(ax=ax2, x='Date', y='Pos_Arr_Delay',color='gray',alpha=0.5,data=df_mean_all,label='ALL',linewidth=2) 
#        ax2.set_ylim(top=20)
        ax2.set_ylabel('Mean Delay')
        ax2.yaxis.set_tick_params(labelsize=8)
    
    ax.set_xlabel('', fontsize=10)
    ax.set_ylabel('Mean Delay',fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4,7,10]))
    ax.set_xlim([datetime.date(2015, 1, 1), datetime.date(2017, 11, 1)])
    ax.set_ylim(bottom=1,top=100)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.legend(labelspacing=0.15,fontsize=10,loc='upper left')
    fig.tight_layout() #reduce margin white space
                  
def iapl_rolling_mean_plot(years, window, airline_list, processed_direc):
    """
    Creates a rolling mean plot of inverse APL for the years specified. 
    Must generate IAPL data first from the atn_graph notebook.

    Parameters
    ----------
    years: list
        Years to plot
    window: int
        The window for the rolling mean
    airline_list: list
        Airlines to plot
    processed_direc: string
        Path to the processed file folder
    
    Returns
    -------
    A matplotlib figure
    
    Notes
    -----
    
    """
    
    #Used to define include_data parameters to the column names
    
    df_airline_all = pd.DataFrame(columns=['Date_Time','Airline','IAPL']) #Initialize a df to take the mean for all years
    #Define figure size
    fig, ax = plt.subplots(figsize=(5.5,2.5))
    
    
    for airline in airline_list:
        for year in years:  
            df_airline = pd.DataFrame()
            file_combined = "%s%s_DD_IAPL.csv" %(processed_direc,year)
            df_year = pd.read_csv(file_combined)
                        
            df_airline['IAPL'] = df_year[airline]
            df_airline['Day_of_Year'] = df_airline.index + 1
            
            #Add a date columne that will use stndard YYYY-MM-DD format
            df_airline['Date_Time'] = pd.to_datetime(year * 1000 + df_airline['Day_of_Year'], format='%Y%j')
            df_airline['Airline'] = airline
            
            #Drop the now unneeded Day_of_Year column
            df_airline = df_airline.drop(columns=['Day_of_Year'])
            
            df_airline_all = pd.concat([df_airline_all,df_airline],ignore_index=True, sort=False)
                   
        df_airline_all['Mean'] = df_airline_all['IAPL'].rolling(window,center=True).mean()
           
    sns.lineplot(ax=ax, x='Date_Time',y='Mean',hue='Airline',style='Airline',data=df_airline_all)   
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4,7,10]))

    ax.set_xlim([datetime.date(2015, 1, 1), datetime.date(2017, 11, 1)])
    ax.set_ylim(top=2.2)
    ax.set_xlabel('',fontsize=10)
    ax.set_ylabel('IAPL',fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc = 'upper left', labelspacing=0.15,
        fontsize=10, handles=handles[1:], labels=labels[1:])
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    fig.tight_layout()
    
def cs_rolling_mean_plot(years, window, airline_list, processed_direc):
    """
    Creates a rolling mean plot of cluster size for the years specified. 
    Must generate IAPL data first from the atn_graph notebook.

    Parameters
    ----------
    years: list
        Years to plot
    window: int
        The window for the rolling mean
    airline_list: list
        Airlines to plot
    processed_direc: string
        Path to the processed file folder
    
    Returns
    -------
    A matplotlib figure
    
    Notes
    -----
    
    """
    
    #Used to define include_data parameters to the column names
    
    df_airline_all = pd.DataFrame(columns=['Date_Time','Airline','Cluster']) #Initialize a df to take the mean for all years
    #Define figure size
    fig, ax = plt.subplots(figsize=(5.5,3))
    
    
    for airline in airline_list:
        for year in years:  
            df_airline = pd.DataFrame()
            file_combined = "%s%s_DD_CS.csv" %(processed_direc,year)
            df_year = pd.read_csv(file_combined)
                        
            df_airline['Cluster_Size'] = df_year[airline]
            df_airline['Day_of_Year'] = df_airline.index + 1
            
            #Add a date columne that will use stndard YYYY-MM-DD format
            df_airline['Date_Time'] = pd.to_datetime(year * 1000 + df_airline['Day_of_Year'], format='%Y%j')
            df_airline['Airline'] = airline
            
            #Drop the now unneeded Day_of_Year column
            df_airline = df_airline.drop(columns=['Day_of_Year'])
            
            df_airline_all = pd.concat([df_airline_all,df_airline],ignore_index=True, sort=False)
                   
        df_airline_all['Mean'] = df_airline_all['Cluster_Size'].rolling(window,center=True).mean()
           
    sns.lineplot(ax=ax, x='Date_Time',y='Mean',hue='Airline',style='Airline',data=df_airline_all)   
    
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4,7,10]))

    ax.set_xlim([datetime.date(2015, 1, 1), datetime.date(2017, 11, 1)])
#    ax.set_ylim(top=2.2)
    ax.set_xlabel('',fontsize=10)
    ax.set_ylabel('Cluster Size',fontsize=10)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.legend(labelspacing=0.15,fontsize=10)
    fig.tight_layout()   
    
    
def iapl_median_lineplot(years, airline_list, processed_direc):
    """
    Creates a plot of the median inverse APL for the years specified with upper and lower quantiles. 
    Must generate IAPL data first from the atn_graph notebook.

    Parameters
    ----------
    years: list
        Years to plot
    airline_list: list
        Airlines to plot
    processed_direc: string
        Path to the processed file folder
    
    Returns
    -------
    A matplotlib figure
    
    Notes
    -----
    
    """
    
    graph_df = pd.DataFrame(columns = ['Airline','Year','Year_Quarter','IAPL'])
    #Define figure size
    fig, ax = plt.subplots(figsize=(5.5,2.5))
              
    for airline in airline_list:
        #Create a df which will take the median and quantile data for each year
        airline_df = pd.DataFrame(columns = ['Year','Quarter','IAPL'])
        
        for year in years:
            df_import = pd.DataFrame()
            file_combined = "%s%s_DD_IAPL.csv" %(processed_direc,year)
            df_raw = pd.read_csv(file_combined)
            
            df_import['IAPL'] = df_raw[airline]
            df_import['Day_of_Year'] = df_import.index + 1
            
            #Add a datetime column
            df_import['Date_Time'] = pd.to_datetime(year * 1000 + df_import['Day_of_Year'], format='%Y%j')
            #Determine the quarter
            df_import['Quarter'] = df_import['Date_Time'].dt.quarter
            
            
            for quarter in range(1,5):
                #Calculate the Quartile Data
                quantile_data = df_import.loc[df_import['Quarter'] == quarter]
                quantile_data = quantile_data.drop(columns=['Day_of_Year'])
                quantile_calc = quantile_data.quantile([0.25,0.5,0.75])
                #Add the corresponding year, quarter, and year_quarter(a single variable which denotes both the year and quarter)
                quantile_calc['Year'] = year
                quantile_calc['Quarter'] = quarter
                quantile_calc['Year_Quarter'] = year+quarter*0.2

                airline_df = pd.concat([airline_df,quantile_calc],ignore_index=True,sort=False)
        airline_df['Airline'] = airline
        graph_df = pd.concat([graph_df,airline_df],ignore_index=True,sort=False)
        
    sns.lineplot(ax=ax, x='Year_Quarter', y='IAPL', hue='Airline', style='Airline', data = graph_df)

#    ax.set_title('IAPL Median with Quantiles from %s - %s' %(years[0],years[-1]))
    ax.set_xlabel('Year',fontsize=10)
    ax.set_ylabel('IAPL',fontsize=10)
    ax.set_xlim(left=2015, right=2018)

    #Set the spacing of the major and minor x-axis ticks.
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.legend(labelspacing=0.15,fontsize=10)
    fig.tight_layout() #reduce margin white space
       
    
def iapl_matthew(years, window, airline_list, processed_direc):
    """
    Creates a rolling mean plot of iapl during the dates of Hurricane Matthew. 
    Must generate IAPL data first from the atn_tools

    Parameters
    ----------
    years: list
        Years to plot
    window: int
        The window for the rolling mean
    airline_list: list
        Airlines to plot
    processed_direc: string
        Path to the processed file folder
    
    Returns
    -------
    A matplotlib figure
    
    Notes
    -----
    
    """
    
    #Used to define include_data parameters to the column names
    
    df_airline_all = pd.DataFrame(columns=['Date_Time','Airline','IAPL']) #Initialize a df to take the mean for all years
    #Define figure size
    fig, ax = plt.subplots(figsize=(5.5,2.5))
    
    
    for airline in airline_list:
        for year in years:  
            df_airline = pd.DataFrame()
            file_combined = "%s%s_DD_IAPL.csv" %(processed_direc,year)
            df_year = pd.read_csv(file_combined)
                        
            df_airline['IAPL'] = df_year[airline]
            df_airline['Day_of_Year'] = df_airline.index + 1
            
            #Add a date columne that will use stndard YYYY-MM-DD format
            df_airline['Date_Time'] = pd.to_datetime(year * 1000 + df_airline['Day_of_Year'], format='%Y%j')
            df_airline['Airline'] = airline
            
            #Drop the now unneeded Day_of_Year column
            df_airline = df_airline.drop(columns=['Day_of_Year'])
            
            df_airline_all = pd.concat([df_airline_all,df_airline],ignore_index=True, sort=False)
                   
        df_airline_all['Mean'] = df_airline_all['IAPL'].rolling(window,center=True).mean()
           
    sns.lineplot(ax=ax, x='Date_Time',y='Mean',hue='Airline',style='Airline',data=df_airline_all)   
    
#    ax.xaxis.set_major_locator(mdates.YearLocator())
#    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4,7,10]))

    ax.set_xlim([datetime.date(2016, 9, 14), datetime.date(2016, 10, 24)])
    ax.set_ylim(top=2.2)
    ax.set_xlabel('',fontsize=10)
    ax.set_ylabel('IAPL',fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.legend(loc='center left',labelspacing=0.15,fontsize=10)
    fig.tight_layout()
        
def cs_matthew(years, window, airline_list, processed_direc):
    """
    Creates a cluster size plot of iapl during the dates of Hurricane Matthew. 
    Must generate CS data first from the atn_tools

    Parameters
    ----------
    years: list
        Years to plot
    window: int
        The window for the rolling mean
    airline_list: list
        Airlines to plot
    processed_direc: string
        Path to the processed file folder
    
    Returns
    -------
    A matplotlib figure
    
    Notes
    -----
    
    """
    
    #Used to define include_data parameters to the column names
    
    df_airline_all = pd.DataFrame(columns=['Date_Time','Airline','Cluster_Size']) #Initialize a df to take the mean for all years
    #Define figure size
    fig, ax = plt.subplots(figsize=(5.5,3))
    
    
    for airline in airline_list:
        for year in years:  
            df_airline = pd.DataFrame()
            file_combined = "%s%s_DD_CS.csv" %(processed_direc,year)
            df_year = pd.read_csv(file_combined)
                        
            df_airline['Cluster_Size'] = df_year[airline]
            df_airline['Day_of_Year'] = df_airline.index + 1
            
            #Add a date columne that will use stndard YYYY-MM-DD format
            df_airline['Date_Time'] = pd.to_datetime(year * 1000 + df_airline['Day_of_Year'], format='%Y%j')
            df_airline['Airline'] = airline
            
            #Drop the now unneeded Day_of_Year column
            df_airline = df_airline.drop(columns=['Day_of_Year'])
            
            df_airline_all = pd.concat([df_airline_all,df_airline],ignore_index=True, sort=False)
                   
        df_airline_all['Mean'] = df_airline_all['Cluster_Size'].rolling(window,center=True).mean()
           
    sns.lineplot(ax=ax, x='Date_Time',y='Mean',hue='Airline',style='Airline',data=df_airline_all)   
    
#    ax.xaxis.set_major_locator(mdates.YearLocator())
#    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4,7,10]))

    ax.set_xlim([datetime.date(2016, 9, 14), datetime.date(2016, 10, 24)])
#    ax.set_ylim(top=2.2)
    ax.set_xlabel('')
    ax.set_ylabel('Cluster Size')
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.legend(labelspacing=0.15)
    fig.tight_layout()
        
    
def hurr_matt(year, airline_list, include_data, processed_direc):
    """
    Creates a rolling mean plot of iapl during the dates of Hurricane Matthew. 
    Must generate IAPL data first from the atn_tools

    Parameters
    ----------
    years: list
        Years to plot
    airline_list: list
        Airlines to plot
    include_data: string
        Specified data filtering type
    processed_direc: string
        Path to the processed file folder
    
    Returns
    -------
    A matplotlib figure
    
    Notes
    -----
    
    """

    maha_dict = {"ADD" : "Arrival_Div_Delay", "ADM":"Arrival_Delay", "DCC":"Delay_Cancel_Count","CC":"Cancel Count", "DD":"Departure_Delay"} #Used to define include_data parameters to the column names
    poi_column = maha_dict[include_data]
    all_plot = 0
    
    
    if 'ALL' in airline_list:
        all_plot = 1  #Change all plot argument so entire network data will be plotted
        airline_list.remove('ALL') #Remove all from the airline list so it is not passed in loop for airlines
        df_whole_network = pd.DataFrame(columns = ['Airline','Year','Year_Quarter',poi_column])
        
        graph_direc = processed_direc + '%s_%s_MDdata_%s.csv' %(year,'ALL',include_data)
        df_md_yearly = pd.read_csv(graph_direc)
        df_md_yearly['Year'] = year
        #Create a column which takes the day of year and maps to a standard pd format;
        df_md_yearly['Date_Time'] = pd.to_datetime(year * 1000 + df_md_yearly['Day_of_Year'], format='%Y%j')
        df_md_yearly = df_md_yearly.drop(columns=['Day_of_Year'])
        
        df_whole_network = pd.concat([df_whole_network,df_md_yearly],ignore_index=True,sort=False)
    
    df_all_airlines = pd.DataFrame(columns=['Airline',poi_column,'Date_Time'])    
    for airline in airline_list:
        graph_direc = processed_direc + '%s_%s_MDdata_%s.csv' %(year,airline,include_data)
        df_md_yearly = pd.read_csv(graph_direc)
        df_md_yearly['Year'] = year
        #Create a column which takes the day of year and maps to a standard pd format;
        df_md_yearly['Date_Time'] = pd.to_datetime(year * 1000 + df_md_yearly['Day_of_Year'], format='%Y%j')
        df_md_yearly = df_md_yearly.drop(columns=['Day_of_Year'])
        
        df_md_yearly['Airline'] = airline
        
        df_all_airlines = pd.concat([df_all_airlines,df_md_yearly],ignore_index=True,sort=False)

    #initialize plot and size
    fig, ax = plt.subplots(figsize=(5.5,2.5))    
    sns.lineplot(ax=ax, data=df_all_airlines, x='Date_Time',y=poi_column, hue='Airline',style='Airline',)
    
    if all_plot == 1:
        ax2 = plt.twinx()
        sns.lineplot(ax=ax2,data=df_whole_network, x='Date_Time', y=poi_column, label='ALL', color='gray',alpha=0.5,linewidth=8.0) 
        ax2.set_ylabel('Mahalabonis distance')
        ax2.set_ylim(bottom=0,top=20)
        ax2.yaxis.set_tick_params(labelsize=8)
    
    #Hard coded range to look at Hurricane Harvey Data
    ax.set_xlim([datetime.date(2016, 9, 14), datetime.date(2016, 10, 24)])
    ax.set_ylim(bottom=0,top=800)
    ax.set_xlabel('',fontsize=10)
    ax.set_ylabel('Mahalabonis distance',fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.legend(loc = 'upper left',fontsize=10)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.legend(labelspacing=0.15)
    
def hurr_harvey(year, airline_list, include_data, processed_direc):
    """
    Creates a rolling mean plot of iapl during the dates of Hurricane Harvey. 
    Must generate IAPL data first from the atn_tools

    Parameters
    ----------
    years: list
        Years to plot
    airline_list: list
        Airlines to plot
    include_data: string
        Specified data filtering type
    processed_direc: string
        Path to the processed file folder
    
    Returns
    -------
    A matplotlib figure
    
    Notes
    -----
    
    """

    maha_dict = {"ADD" : "Arrival_Div_Delay", "ADM":"Arrival_Delay", "DCC":"Delay_Cancel_Count","CC":"Cancel Count", "DD":"Departure_Delay"} #Used to define include_data parameters to the column names
    poi_column = maha_dict[include_data]
    all_plot = 0
    
    
    if 'ALL' in airline_list:
        all_plot = 1  #Change all plot argument so entire network data will be plotted
        airline_list.remove('ALL') #Remove all from the airline list so it is not passed in loop for airlines
        df_whole_network = pd.DataFrame(columns = ['Airline','Year','Year_Quarter',poi_column])
        
        graph_direc = processed_direc + '%s_%s_MDdata_%s.csv' %(year,'ALL',include_data)
        df_md_yearly = pd.read_csv(graph_direc)
        df_md_yearly['Year'] = year
        #Create a column which takes the day of year and maps to a standard pd format;
        df_md_yearly['Date_Time'] = pd.to_datetime(year * 1000 + df_md_yearly['Day_of_Year'], format='%Y%j')
        df_md_yearly = df_md_yearly.drop(columns=['Day_of_Year'])
        
        df_whole_network = pd.concat([df_whole_network,df_md_yearly],ignore_index=True,sort=False)
    
    df_all_airlines = pd.DataFrame(columns=['Airline',poi_column,'Date_Time'])    
    for airline in airline_list:
        graph_direc = processed_direc + '%s_%s_MDdata_%s.csv' %(year,airline,include_data)
        df_md_yearly = pd.read_csv(graph_direc)
        df_md_yearly['Year'] = year
        #Create a column which takes the day of year and maps to a standard pd format;
        df_md_yearly['Date_Time'] = pd.to_datetime(year * 1000 + df_md_yearly['Day_of_Year'], format='%Y%j')
        df_md_yearly = df_md_yearly.drop(columns=['Day_of_Year'])
        
        df_md_yearly['Airline'] = airline
        
        df_all_airlines = pd.concat([df_all_airlines,df_md_yearly],ignore_index=True,sort=False)
    #initialize plot and size
    fig, ax = plt.subplots(figsize=(10,5.5))    
    sns.lineplot(ax=ax, data=df_all_airlines, x='Date_Time',y=poi_column, hue='Airline')
    
    if all_plot == 1:
        ax2 = plt.twinx()
        sns.lineplot(ax=ax2,data=df_whole_network, x='Date_Time', y=poi_column, label='ALL', color='gray',alpha=0.5,linewidth=8.0) 
        ax2.set_ylabel('Mahalabonis-Distance')
        ax2.set_ylim(bottom=0,top=30)
    
    #Hard coded range to look at Hurricane Harvey Data
    ax.set_xlim([datetime.date(2017, 8, 3), datetime.date(2017, 9, 17)])
    ax.set_ylim(bottom=0,top=1200)
    ax.set_xlabel('')
    ax.set_ylabel('Mahalabonis-Distance')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    ax.legend(loc = 'upper left')
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    
def ran_remove_shaded(year_list, airline_list, processed_direc, graph_direc):
    """
    Creates a shaded graphs for the pure network metric random removals

    Parameters
    ----------
    year_list: list
        Years to plot
    airline_list: list
        Airlines to plot
    processed_direc: string
        Path to the processed data folder
    graph_direc: string
        Path to the graph data folder
    
    Returns
    -------
    None
    
    Notes
    -----
    Saves figures to the /data/graph folder.
    
    """
    
    IAPL_df_all = pd.DataFrame(columns = ['Year','Airline','IAPL'])
    CS_df_all = pd.DataFrame(columns = ['Year','Airline','Cluster_Size'])
    AC_df_all = pd.DataFrame(columns = ['Year','Airline','AC'])    
    for airline in airline_list:
        script_dir = os.path.dirname(os.getcwd())
        CS_path = "%s%s_CSR.csv" %(processed_direc,airline)
        CS_file = os.path.join(script_dir,CS_path)
        CS_df = pd.read_csv(CS_file)

        IAPL_path = "%s%s_IAPLR.csv" %(processed_direc,airline)
        IAPL_file = os.path.join(script_dir,IAPL_path)
        IAPL_df = pd.read_csv(IAPL_file)

        AC_path = "%s%s_ACR.csv" %(processed_direc,airline)
        AC_file = os.path.join(script_dir,AC_path)
        AC_df = pd.read_csv(AC_file)

        CS_df_airline = pd.DataFrame(columns = ['Year','Airline','Cluster_Size'])
        CS_year_df = pd.DataFrame()

        IAPL_df_airline = pd.DataFrame(columns = ['Year','Airline','IAPL'])
        IAPL_year_df = pd.DataFrame()

        AC_df_airline = pd.DataFrame(columns = ['Year','Airline','AC'])
        AC_year_df = pd.DataFrame()

        col = 0
        for year in year_list:
            CS_year_df['Cluster_Size'] = CS_df.iloc[:,col]
            CS_quant_calc = CS_year_df.quantile([0.25,0.5,0.75])
            CS_quant_calc['Year'] = year
            CS_df_airline = pd.concat([CS_df_airline,CS_quant_calc],ignore_index=True)

            IAPL_year_df['IAPL'] = IAPL_df.iloc[:,col]
            IAPL_quant_calc = IAPL_year_df.quantile([0.25,0.5,0.75])
            IAPL_quant_calc['Year'] = year
            IAPL_df_airline = pd.concat([IAPL_df_airline,IAPL_quant_calc],ignore_index=True)

            AC_year_df['AC'] = AC_df.iloc[:,col]
            AC_quant_calc = AC_year_df.quantile([0.5,0.5,0.5])
            AC_quant_calc['Year'] = year
            AC_df_airline = pd.concat([AC_df_airline,AC_quant_calc],ignore_index=True)

            col = col + 1
        CS_df_airline['Airline'] = airline
        CS_df_all = pd.concat([CS_df_all,CS_df_airline],ignore_index = True)

        IAPL_df_airline['Airline'] = airline
        IAPL_df_all = pd.concat([IAPL_df_all,IAPL_df_airline],ignore_index = True)

        AC_df_airline['Airline'] = airline
        AC_df_all = pd.concat([AC_df_all,AC_df_airline],ignore_index = True)


    plt.figure(1,figsize=(2.8,2.0))
    ax1 = sns.lineplot(data=CS_df_all, x = 'Year', y = 'Cluster_Size', hue='Airline', style='Airline', marker = 'o')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel('Year')
    plt.ylabel('Cluster Size')
    plt.legend(airline_list,fontsize=10,labelspacing=0.15)
    plt.tight_layout()
    plt.savefig('%sShaded_CS.pdf'%(graph_direc,))

    plt.figure(2,figsize=(2.8,2.0))
    ax2 = sns.lineplot(data=IAPL_df_all, x = 'Year', y = 'IAPL', hue='Airline', style='Airline', marker = 'o')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel('Year')
    plt.ylabel('IAPL')
    plt.legend(airline_list,fontsize=10,labelspacing=0.15)
    plt.tight_layout()
    plt.savefig('%sShaded_IAPL.pdf'%(graph_direc,))

    plt.figure(3,figsize=(2.8,2.0))
    ax3 = sns.lineplot(data=AC_df_all, x = 'Year', y = 'AC', hue='Airline', style='Airline', marker = 'o')
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel('Year')
    plt.ylabel('Algebraic Connectivity')
    plt.legend(airline_list,fontsize=10,labelspacing=0.15) 
    plt.tight_layout()
    plt.savefig('%sShaded_AC.pdf'%(graph_direc,))

    plt.show()

def us_map_plot(
    db_path,
    processed_direc,
    year,
    graph_direc,
    airline,
    node_weights,
    edge_weights,
    graph_name='',
    show_plot=True):
    """
    Displays a visualization of weighted nodes (based off removal data), weighted edges (based off weighted data) for a given airline and year

    Parameters
    ----------
    year: int
        Year of selected data
    airline: string
        Airline to get data from
    include_data: string
        Type of airline data to query from csv
    node_weights: dict
        Dictionary with keys as airport names and values as weights for the node 
    edge_weights: pandas DataFrame object
        Pandas Dataframe with three columns
            Origin - Original airport names
            Destination - Destination airport names
            Weight - Weight for the edge/route
    
    Returns
    -------
    
    Notes
    -----
    Uses the Basemap library which must be externally installed in order for the visualization to be possible
    
    """
    # plt.figure(figsize=(5.5,3),dpi=300)
    plt.figure(figsize=(3.0,2.1),dpi=300)

    conn = sqlite3.connect(db_path)
    
    origin_sql = "SELECT * FROM airportCoords WHERE IATA IN ('%s')" %("','".join(node_weights.keys()))

    routes_us = pd.read_sql(origin_sql,conn)

    graph = nx.from_pandas_edgelist(
        edge_weights,
        source = 'Origin',
        target = 'Destination',
        edge_attr = 'Weight',
        create_using = nx.DiGraph())  

    m = Basemap(projection='merc',llcrnrlon=-125,llcrnrlat=25,urcrnrlon=-65,
            urcrnrlat=50, lat_ts=0, resolution='h',suppress_ticks=True)

    m.drawstates(linewidth=0.4)
    m.drawcountries(linewidth=0.4)
    m.drawlsmask(land_color='white',ocean_color='white',lakes=True)
    m.drawcoastlines(linewidth=0.4)

    # Convert lat/long coordinates into basemap coordinates
    mx, my = m(routes_us['long'].values, routes_us['lat'].values)
    pos = {}

    for count, elem in enumerate(routes_us['IATA'].values):
        pos[elem] = (mx[count], my[count])

    # Create a Green to Red colormap to use on weighted edges
    # https://stackoverflow.com/questions/38246559/how-to-create-a-heat-map-in-python-that-ranges-from-green-to-red 
    cdict = {'red':  ((0.0, 0.0, 0.0),   # no red at 0
                (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1

    'green': ((0.0, 0.8, 0.8),   # set to 0.8 so its not too bright at 0
                (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                (1.0, 0.0, 0.0)),  # no green at 1

    'blue':  ((0.0, 0.0, 0.0),   # no blue at 0
                (0.5, 1.0, 1.0),   # all channels set to 1.0 at 0.5 to create white
                (1.0, 0.0, 0.0))   # no blue at 1
    }

    # Create the colormap using the dictionary
    GnRd = matplotlib.colors.LinearSegmentedColormap('GnRd', cdict)

    node_weights = [node_weights[node] for node in graph.nodes()]
    #Scale node sizes between 0 and 30 for plotting
    node_weight_norm = [
        30*((i - min(node_weights))/ 
        (max(node_weights)-min(node_weights))) + 1
        for i in node_weights]

    # draw the weighted edges and nodes
    nx.draw_networkx_nodes(
        G = graph,
        pos = pos,
        node_list = graph.nodes(),
        node_color=node_weights,
        edgecolors='None',
        cmap=plt.cm.autumn_r,
        alpha = 0.8,
        node_size = node_weight_norm)

    nx.draw_networkx_edges(
        G = graph, 
        pos = pos,
        edge_color=edge_weights['Weight'],
        edge_cmap=plt.cm.RdYlGn,
        alpha=0.8,
        arrows = False)   
    
    plt.tight_layout()
    
    save_path = "%s%s_%s_%s_map.pdf"%(graph_direc,airline,year,graph_name)
    plt.savefig(save_path,facecolor='white')

    print("Map saved to %s" %(save_path,))
    if show_plot:
        plt.show()
       
def md_median_plot(years, airline, include_data, processed_direc):
    """
    Parameters
    ----------
    years: list
        List of years to plot.
    airline: str
        The airline whose data to plot.
    include_data: string
        Specify what kind of data to include in processed flight data. See drop_flights in M-D File. Possible parameters are:
            CC: Cancellations only
            ADD: Arrival delays including diversions
            ADM: Purely arrival delays excluding cancellations or diversions
            DCC: Combined delay. If arrival delay is greater than a set threshold, the flight is considered cancelled
            DD: Departure delays. Does not include cancelled or diverted flights.
    processed_direc: string
        Location of the folder for the processed M-D data csv to be retrieved from
        
    Returns
    -------
    Creates a graph for the M-D data or multiple airlines for a specified year
    
    Notes
    -----
    The M-D data must be created first. See the "mahalanobis_distance" function in "query_atn_db".
    """
       
    
    maha_dict = {"ADD" : "Arrival_Div_Delay", "ADM":"Arrival_Delay", "DCC":"Delay_Cancel_Count","CC":"Cancel Count", "DD":"Departure_Delay"} #Used to define include_data parameters to the column names

    md_medians = []
    
    for year in years:
        graph_direc = processed_direc + '%s_%s_MDdata_%s.csv' %(year,airline,include_data)
        df_graph = pd.read_csv(graph_direc)
        md_medians.append(np.nanmedian(df_graph.loc[:,maha_dict[include_data]]))
        
    plt.plot(years,md_medians)
    plt.title('Median of Mahalanobis-Distance for %s from %s - %s' %(airline,years[0],years[-1]))
    
def airport_coord_query(db_path, file, airline):
    """
    Retrieves all the airports that appear in an airline's network for a given year.

    Parameters
    ----------
    file: int
        Year of selected data
    airline: string
        Airline to get data from
    
    Returns
    -------
    Returns a data frame containing the IATA, latitude and longitude values of all the airports. 
    
    Notes
    -----
    
    """

    engine = create_engine('sqlite:///%s'%(db_path,))
    conn = sqlite3.connect(db_path)

    labels = ['Origin', 'x', 'y']
    
    df_airports = analysis.raw_query(db_path,file,airline)

    origin_airports = df_airports["Origin_Airport_Code"].tolist()

    destination_airports = df_airports["Destination_Airport_Code"].tolist()

    airport_list = set(origin_airports + destination_airports) # Set gets unique

    origin_sql = "SELECT * FROM airportCoords WHERE IATA IN ('%s')" %("','".join(airport_list))

    df_origin = pd.read_sql(origin_sql,engine)

    lats_o = []
    longs_o = []
    for airport in airport_list:
        for j in range(len(df_origin.IATA)):
            if airport == df_origin.iloc[j].IATA:
                lats_o.append(df_origin.iloc[j].lat)
                longs_o.append(df_origin.iloc[j].long)

    df = pd.DataFrame(list(zip(airport_list,lats_o,longs_o)), columns=['IATA','lat','long'])
    return(df)