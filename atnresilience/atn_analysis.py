import os
import copy
import calendar

import pandas as pd
import numpy as np
import sqlite3
import scipy.stats as st

# from create_atn_db import DBLoader
import db_tools as DBTools

class AnalysisQueries(DBTools.DBQueries):
    def __init__(self,db_path = None, raw_path = None):
        """
        Initialize common parameters used for the database and inputs.
        If none given, they are defaulted to their expected locations within the repository.
        Default Database Location: /data/processed/atn_db.sqlite
        Default Raw Data Folder: /data/raw/

        Parameters
        ----------
        db_path: string
            Path to the database
        raw_path: string
            Path to folder containing raw data
        
        Returns
        ----------
        None 

        """
        root_dir = os.path.abspath(os.path.join(os.getcwd(),".."))

        if db_path:
            self.db_path = db_path
        else:
            self.db_path = os.path.join(root_dir,'data','processed','atn_db.sqlite')
        
        if raw_path:
            self.raw_path = raw_path
        else:
            self.raw_path = os.path.join(root_dir,'data','raw','')

    def airline_query(self, year, airline, include_data='ADM'):
        """
        Creates a pandas dataframe with a list of airports that are relevant after dropping 
        and corresponding columns of data based on the include_data parameter.
        Get a list of airports which have flights every day then perform summing or averaging on
        the relevant metric based on type
        
        Parameters
        ----------
        db_path: string
            Path to the location of the atn database
        year: int
            Year of data you would like to process.
        airline: string
            Airline identified code (ie 'AA','DL','UA')
            Use 'ALL' to retrieve for all airlines
        include_data: string
            Defaults to ADM
            Specify what kind of data to include in processed flight data. See drop_flights in M-D File. Possible parameters are:
                CC: Cancellations only
                ADD: Arrival delays including diversions
                ADM: Purely arrival delays excluding cancellations or diversions
                DCC: Combined delay. If arrival delay is greater than a set threshold, the flight is considered cancelled
                DD: Departure delays. Does not include cancelled or diverted flights.
        
        Returns
        -------
        dataframe
            Dataframe grouped by origin or destination airport with key metric columns summed or averaged depending
            on include_data type specified
        """
        include_filter = {"ADD" : "Can_Status != 1 AND Combined_Arr_Delay IS NOT NULL", 
            "ADM":"Can_Status != 1 AND Div_Status != 1", 
            "CC":"", 
            "DD":"Can_Status != 1 AND Dep_Delay IS NOT NULL", 
            "DCC": "Combined_Arr_Delay IS NOT NULL"} 
        if airline == 'ALL':
            # Return all airlines in SQL statement
            airline = '%'
        if include_data == 'ADD':
            airline_sql = '''
            SELECT Flight_Date, Day_Of_Year AS Day, Destination_Airport_Code,
            AVG(Combined_Arr_Delay) AS Combined_Arr_Delay
            FROM atn_performance
            WHERE Destination_Airport_Code IN (
                SELECT Destination_Airport_Code
                FROM atn_performance
                WHERE Year = :year
                AND Unique_Carrier_ID LIKE :airline
                AND Can_Status != 1 
                AND Combined_Arr_Delay IS NOT NULL 
                GROUP BY Destination_Airport_Code
                HAVING COUNT(DISTINCT(Day_Of_Year)) = :days_in_year
                )
            AND Year = :year
            AND Unique_Carrier_ID LIKE :airline
            AND Can_Status != 1 
            AND Combined_Arr_Delay IS NOT NULL 
            GROUP BY Flight_Date, Destination_Airport_Code
            '''

        if include_data == 'ADM':
            airline_sql = '''
            SELECT Flight_Date, Day_Of_Year AS Day, Destination_Airport_Code,
            AVG(Pos_Arr_Delay) AS Pos_Arr_Delay
            FROM atn_performance
            WHERE Destination_Airport_Code IN (
                SELECT Destination_Airport_Code
                FROM atn_performance
                WHERE Year = :year
                AND Unique_Carrier_ID LIKE :airline
                AND Can_Status != 1 
                AND Div_Status != 1 
                GROUP BY Destination_Airport_Code
                HAVING COUNT(DISTINCT(Day_Of_Year)) = :days_in_year
                )
            AND Year = :year
            AND Unique_Carrier_ID LIKE :airline
            AND Can_Status != 1 
            AND Div_Status != 1 
            GROUP BY Flight_Date, Destination_Airport_Code
            '''

        elif include_data == 'CC':
            airline_sql = '''
            SELECT Flight_Date, Day_Of_Year AS Day, Origin_Airport_Code,
            SUM(Can_Status) AS Can_Status
            FROM atn_performance
            WHERE Origin_Airport_Code IN (
                SELECT Origin_Airport_Code
                FROM atn_performance
                WHERE Year = :year
                AND Unique_Carrier_ID LIKE :airline
                GROUP BY Origin_Airport_Code
                HAVING COUNT(DISTINCT(Day_Of_Year)) = :days_in_year
                )
            AND Year = :year
            AND Unique_Carrier_ID LIKE :airline
            GROUP BY Flight_Date, Origin_Airport_Code
            '''

        elif include_data == 'DD':
            airline_sql = '''
            SELECT Flight_Date, Day_Of_Year AS Day, Origin_Airport_Code,
            AVG(Dep_Delay) AS Dep_Delay
            FROM atn_performance
            WHERE Origin_Airport_Code IN (
                SELECT Origin_Airport_Code
                FROM atn_performance
                WHERE Year = :year
                AND Unique_Carrier_ID LIKE :airline
                AND Can_Status != 1 
                AND Combined_Arr_Delay IS NOT NULL
                GROUP BY Origin_Airport_Code
                HAVING COUNT(DISTINCT(Day_Of_Year)) = :days_in_year
                )
            AND Year = :year
            AND Unique_Carrier_ID LIKE :airline
            AND Can_Status != 1 
            AND Combined_Arr_Delay IS NOT NULL
            GROUP BY Flight_Date, Origin_Airport_Code
            '''

        elif include_data == 'DCC':
            airline_sql = '''
            SELECT Flight_Date, Day_Of_Year, Destination_Airport_Code, Can_Status,
            SUM(
                CASE 
                    WHEN Combined_Arr_Delay > 14.99 THEN 1
                    WHEN Combined_Arr_Delay <= 14.99 THEN 0
                END 
                )AS Del_Count
            FROM atn_performance
            WHERE Destination_Airport_Code IN (
                SELECT Destination_Airport_Code,
                COUNT(DISTINCT(Day_Of_Year)) DAY_COUNT
                FROM atn_performance
                GROUP BY Destination_Airport_Code
                WHERE Year = :year
                AND Unique_Carrier_ID LIKE :airline
                AND Combined_Arr_Delay IS NOT NULL
                HAVING COUNT(DISTINCT(Day_Of_Year)) = :days_in_year
                )
            AND Year = :year
            AND Unique_Carrier_ID LIKE :airline
            AND Combined_Arr_Delay IS NOT NULL
            GROUP BY Flight_Date, Destination_Airport_Code
            '''

        else:
            return('Not a valid include_data type')

        if calendar.isleap(year) == 1:
            days_in_year = 366
        else:
            days_in_year = 365

        airline_df = self.query_to_df(
            airline_sql,
            params={'year' : year, 
                'days_in_year' : days_in_year,
                'airline' : airline})

        if include_data == 'DCC':
            airline_df['Del_Can_Count'] = airline_df['Del_Count'] + airline_df['Can_Status']

        return(airline_df)

    def airport_stats_query(self, year, airline, include_data='ADM'):
        """
        Creates a pandas dataframe with a list of airports that are relevant after dropping 
        and corresponding columns of data based on the include_data parameter.
        Get a list of airports which have flights every day then perform summing or averaging on
        the relevant metric based on type
        
        Parameters
        ----------
        db_path: string
            Path to the location of the atn database
        year: int
            Year of data you would like to process.
        airline: string
            Airline identified code (ie 'AA','DL','UA')
            Use 'ALL' to retrieve for all airlines
        include_data: string
            Defaults to ADM
            Specify what kind of data to include in processed flight data. See drop_flights in M-D File. Possible parameters are:
                CC: Cancellations only
                ADD: Arrival delays including diversions
                ADM: Purely arrival delays excluding cancellations or diversions
                DCC: Combined delay. If arrival delay is greater than a set threshold, the flight is considered cancelled
                DD: Departure delays. Does not include cancelled or diverted flights.
        
        Returns
        -------
        dataframe
            Dataframe grouped by origin or destination airport with key metric columns summed or averaged depending
            on include_data type specified
        """
        if airline == 'ALL':
            # Return all airlines in SQL statement
            airline = '%'

        if include_data == 'ADD':
            airline_sql = '''
            SELECT Destination_Airport_Code,
            COUNT(DISTINCT(Day_Of_Year)) AS Day_Count
            FROM atn_performance
            WHERE Year = :year
            AND Unique_Carrier_ID LIKE :airline
            AND Can_Status != 1 
            AND Combined_Arr_Delay IS NOT NULL 
            GROUP BY Destination_Airport_Code
            '''

        if include_data == 'ADM':
            airline_sql = '''
            SELECT Destination_Airport_Code,
            COUNT(DISTINCT(Day_Of_Year)) AS Day_Count
            FROM atn_performance
            WHERE Year = :year
            AND Unique_Carrier_ID LIKE :airline
            AND Can_Status != 1 
            AND Div_Status != 1 
            GROUP BY Destination_Airport_Code
            '''

        elif include_data == 'CC':
            airline_sql = '''
            SELECT Origin_Airport_Code,
            COUNT(DISTINCT(Day_Of_Year)) AS Day_Count
            FROM atn_performance
            WHERE Year = :year
            AND Unique_Carrier_ID LIKE :airline
            GROUP BY Origin_Airport_Code
            '''

        elif include_data == 'DD':
            airline_sql = '''
            SELECT Origin_Airport_Code,
            COUNT(DISTINCT(Day_Of_Year)) AS Day_Count
            FROM atn_performance
            WHERE Year = :year
            AND Unique_Carrier_ID LIKE :airline
            AND Can_Status != 1 
            AND Combined_Arr_Delay IS NOT NULL
            GROUP BY Origin_Airport_Code
            '''

        elif include_data == 'DCC':
            airline_sql = '''
            SELECT Destination_Airport_Code,
            COUNT(DISTINCT(Day_Of_Year)) AS Day_Count,
            FROM atn_performance
            GROUP BY Destination_Airport_Code
            WHERE Year = :year
            AND Unique_Carrier_ID LIKE :airline
            AND Combined_Arr_Delay IS NOT NULL

            '''

        else:
            return('Not a valid include_data type')

        stats_df = self.query_to_df(
            airline_sql,
            params={'year' : year,
                'airline' : airline})

        if include_data == 'DCC':
            stats_df['Del_Can_Count'] = stats_df['Del_Count'] + stats_df['Can_Status']

        return(stats_df)

class MahalanobisDistance(object):
    def __init__(self, year, airline, db_path = None, processed_direc = None, include_data = "ADM"):

        root_dir = os.path.abspath(os.path.join(os.getcwd(),".."))

        if db_path:
            self.db_path = db_path
        else:
            self.db_path = os.path.join(root_dir,'data','processed','atn_db.sqlite')

        if processed_direc:
            self.processed_direc = processed_direc
        else:
            self.processed_direc = os.path.join(root_dir,'data','processed','')
        self.year = year
        self.airline = airline
        self.include_data = include_data

    def mahalanobis_distance(self):
        """
        Creates a csv of M-D data for each day depending on the data sorting specified (include_data)
        Done on a year-basis for a specific airline or the whole network
        Parameters
        ----------
        db_path: string
            Path to the location of the atn database
        year: int
            Year of data you would like to process.
        airline: string
            Code for the airline to process data for
            Use '' to retrieve for all airlines
        include_data: string
            Specify what kind of data to include in processed flight data. See drop_flights in M-D File. Possible parameters are:
                CC: Cancellations only
                ADD: Arrival delays including diversions
                ADM: Purely arrival delays excluding cancellations or diversions
                DCC: Combined delay. If arrival delay is greater than a set threshold, the flight is considered cancelled
                DD: Departure delays. Does not include cancelled or diverted flights.
        processed_direc: string
            Location of the folder for the processed MD csv to be created
        
        Returns
        -------
        file: csv
            Returns as csv in the /data/processed folder with  the naming year_airline_MDdata_includedata.csv
        
        
        Notes
        -----
        The atn db must be created first. Please see create_atn_db.
        
        """
        querries = AnalysisQueries()
        #Call the airline_query function below to retrieve the airports to use for M-D calculation
        by_dest = querries.airline_query(self.year, self.airline, include_data = self.include_data)

        #If no airline is specified, call the airline ALL to be used for file naming later. This needs to be done after calling airline_query so we do not pass the wrong airline to that function
        if self.airline == '':
            self.airline = 'ALL'

    ## The following is equivalent to Keshav's original mahalanobis_distance script
        #Calculate the of dates in a year that correspond to each day of week. IE Monday will have a list of day of week, Tuesday will have a list of day of week, etc...
        
        if calendar.isleap(self.year) == 1:
            no_of_days = 366
        else:
            no_of_days = 365
        day_list_all = []
        
        #day_list specifies which day of year corresponds to which day of the week
        #ie, Monday = 3,10,17,24,etc
        for j in range(1, 8):
            day_list_all.append([])
            for i in range(j, no_of_days + 1, 7):
                day_list_all[-1].append(i)
        
        #initialize a blank df to use in calculation
        df_maha_com = pd.DataFrame()
        #Loop through each day of the week
        for i in range(7):
            day_list = day_list_all[i]
            df_maha = copy.deepcopy(by_dest)
            df_maha = df_maha[df_maha['Day'].isin(day_list)].dropna()
            a_t = pd.DataFrame()
            m_t = pd.DataFrame()
            n = len(day_list)
            #Loop through each day
            for day in day_list:
                df_not_day = df_maha[df_maha.Day != day]
                df_for_day = df_maha[df_maha.Day == day]
                if self.include_data == "ADD":
                    a_t[int((day - 1) / 7)] = df_for_day.Combined_Arr_Delay.reset_index(drop=True)
                    df_mean = df_not_day['Combined_Arr_Delay'].groupby([df_not_day['Destination_Airport_Code']]).mean().reset_index()
                    m_t[int((day - 1) / 7)] = df_mean["Combined_Arr_Delay"]
                if self.include_data == "ADM":
                    a_t[int((day - 1) / 7)] = df_for_day.Pos_Arr_Delay.reset_index(drop=True)
                    df_mean = df_not_day['Pos_Arr_Delay'].groupby([df_not_day['Destination_Airport_Code']]).mean().reset_index()
                    m_t[int((day - 1) / 7)] = df_mean["Pos_Arr_Delay"]
                if self.include_data == "CC":
                    a_t[int((day - 1) / 7)] = df_for_day.Pos_Arr_Delay.reset_index(drop=True)
                    df_mean = df_not_day['Pos_Arr_Delay'].groupby([df_not_day['Destination_Airport_Code']]).mean().reset_index()
                    m_t[int((day - 1) / 7)] = df_mean["Pos_Arr_Delay"]
                if self.include_data == "DCC":
                    a_t[int((day - 1) / 7)] = df_for_day.Del_Can_Count.reset_index(drop=True)
                    df_mean = df_not_day['Del_Can_Count'].groupby([df_not_day['Destination_Airport_Code']]).mean().reset_index()
                    m_t[int((day - 1) / 7)] = df_mean["Del_Can_Count"]
                if self.include_data == "DD":
                    a_t[int((day - 1) / 7)] = df_for_day.Dep_Delay.reset_index(drop=True)
                    df_mean = df_not_day['Dep_Delay'].groupby([df_not_day['Origin_Airport_Code']]).mean().reset_index()
                    m_t[int((day - 1) / 7)] = df_mean["Dep_Delay"]
            mean_size = int(len(m_t))
            df_sum = pd.DataFrame(np.zeros((mean_size, mean_size)))
            #Loop through each day for that day of week
            for k in range(n):
                a_mat = a_t[k].values
                a_tt = pd.DataFrame(np.outer(a_mat, a_mat))
                df_sum = df_sum.add(a_tt.div(n - 2))
            dict_values = {}
            
            #Loop through each day for that day of week
            for l in range(n):
                # Get products of value and mean vectors and their transpose
                a_mat = a_t[l].values
                a_tt = pd.DataFrame(np.outer(a_mat, a_mat))
                m_mat = m_t[l].values
                m_tt = pd.DataFrame(np.outer(m_mat, m_mat))

                # Sum in the covariance matrix
                sigma = df_sum.subtract(a_tt.div(n - 2).subtract(m_tt.div((2 - n) / (n - 1) ** 2)))
                sigma.fillna(value=0,inplace=True)
                inv = np.linalg.pinv(sigma.values)
                # print(inv)
                diff = (a_t[l] - m_t[l])[:, None]
                maha_t = diff.transpose().dot(inv).dot(diff)
                dict_values[l] = [maha_t[0, 0]]
                df_mahalanobis = pd.DataFrame(dict_values).T

            day_list_series = pd.Series(day_list)
            df_mahalanobis['Day'] = day_list_series.values
            df_mahalanobis.set_index('Day', inplace=True)
            df_maha_com = pd.concat([df_maha_com, df_mahalanobis])
            
        df_maha_com.sort_index(inplace=True)
        df_maha_com[0] = np.sqrt(df_maha_com[0])

        maha_dict = {"ADD" : "Arrival_Div_Delay", "ADM":"Arrival_Delay", "DCC":"Delay_Cancel_Count","CC":"Cancel Count", "DD":"Departure_Delay"} #Used to define column names in the final output
        
        df_maha_com.columns = [maha_dict[self.include_data]] #Change the column name to the correct parameter
        df_maha_com.insert(loc=0, column = 'Day_of_Year', value = df_maha_com.index)

        export_filename = os.path.join(self.processed_direc,'{}_{}_MDdata_{}.csv'.format(self.airline,self.year,self.include_data))
        # file_combined = "%s%s_%s_MDdata_%s.csv" %(self.processed_direc,self.year,self.airline,self.include_data)
        df_maha_com.to_csv(export_filename, index=False)
        print('%s M-D Data created at %s' %(self.year,export_filename))

    
def z_score(db_path, year, airline, include_data, processed_direc):
    """
    Calculate the Z-score for each airport on each day of the year. For each airport at each day, the Z-score
    is calculated relative to the mean and variance of the airport and values in the other days of week that the
    day of year corresponds to.

    Parameters
    ----------
    db_path: string
        Path to the location of the atn database
    year: int
        Year of data you would like to process.
    airline: string
        Code for the airline to process data for
        Use '' to retrieve for all airlines
    include_data: string
        Specify what kind of data to include in processed flight data. See drop_flights in M-D File. Possible parameters are:
            CC: Cancellations only
            ADD: Arrival delays including diversions
            ADM: Purely arrival delays excluding cancellations or diversions
            DCC: Combined delay. If arrival delay is greater than a set threshold, the flight is considered cancelled
            DD: Departure delays. Does not include cancelled or diverted flights.
    processed_direc: string
        Location of the folder for the processed data csv to be created
        
    Returns
    -------
    file: csv
        Returns a csv in /data/processed with the name year_airline_Zdata_includedata.csv
    
    Notes
    -----
    The atn db must be created first. Please see create_atn_db.
    
    """
    querries = AnalysisQueries()
    #Call the airline_query function below to retrieve the airlines to be used for analysis
    df_score_ori = querries.airline_query(year,airline)
    print('Airport data imported from Query')
    
    #If no airline is specified, call the airline ALL to be used for file naming later. This needs to be done after calling airline_query so we do not pass the wrong airline to the function
    if airline == '':
        airline = 'ALL'
        
    if calendar.isleap(year) == 1:
        no_of_days = 366
    else:
        no_of_days = 365

    day_list_all = []
    for j in range(1, 8):
        day_list_all.append([])
        for i in range(j, no_of_days + 1, 7):
            day_list_all[-1].append(i)

    print("Calculating metrics")
    
    #Depending on the sorting used, find a list of the Origin or Destination airports that appear
    if include_data in ["CC","DD",]:
        list_airport = df_score_ori['Origin_Airport_Code'].unique()
    else:
        list_airport = df_score_ori['Destination_Airport_Code'].unique()
        
    c = 0
    df_zscore_all = pd.DataFrame()
    #Loop through each airport
    for d in list_airport:
        c = c+1
        df_score_all = copy.deepcopy(df_score_ori)
        
        if include_data in ["CC","DD",]:
            df_score_all = df_score_all[df_score_all['Origin_Airport_Code'] == d]
        else:
            df_score_all = df_score_all[df_score_all['Destination_Airport_Code'] == d]
        df_zscore_com = pd.DataFrame()
        
        #Loop through each day of the week (Mon.,Tues,Wed.,etc..)
        for i in range(7):
            day_list = day_list_all[i]
            df_score = copy.deepcopy(df_score_all)
            df_score = df_score[df_score['Day'].isin(day_list)].dropna()

            x_t = pd.DataFrame()    # Random variable - Parameter of interest for that day\
            m_t = pd.DataFrame()    # Time dependent mean - Mean of parameter for that day ( due to dropping that day)
            n = len(day_list)
            
            # Calculating means for each specific day of the year that day of week appears
            for day in day_list:
                df_not_day = df_score[df_score.Day != day]
                df_for_day = df_score[df_score.Day == day]
                if include_data == "CC":
                    x_t[int((day - 1) / 7)] = df_for_day.Can_Status.reset_index(drop=True)
                    df_mean = df_not_day['Can_Status'].groupby([df_not_day['Origin_Airport_Code']]).mean().reset_index()
                    m_t[int((day - 1) / 7)] = df_mean["Can_Status"]

                if include_data == "ADD":
                    x_t[int((day - 1) / 7)] = df_for_day.Combined_Arr_Delay.reset_index(drop=True)
                    df_mean = df_not_day['Combined_Arr_Delay'].groupby([df_not_day['Destination_Airport_Code']]).mean().reset_index()
                    m_t[int((day - 1) / 7)] = df_mean["Combined_Arr_Delay"]
                if include_data == "ADM":
                    x_t[int((day - 1) / 7)] = df_for_day.Pos_Arr_Delay.reset_index(drop=True)
                    df_mean = df_not_day['Pos_Arr_Delay'].groupby([df_not_day['Destination_Airport_Code']]).mean().reset_index()
                    m_t[int((day - 1) / 7)] = df_mean["Pos_Arr_Delay"]
                if include_data == "DCC":
                    x_t[int((day - 1) / 7)] = df_for_day.Del_Can_Count.reset_index(drop=True)
                    df_mean = df_not_day['Del_Can_Count'].groupby([df_not_day['Destination_Airport_Code']]).mean().reset_index()
                    m_t[int((day - 1) / 7)] = df_mean["Del_Can_Count"]
                if include_data == "DD":
                    x_t[int((day - 1) / 7)] = df_for_day.Dep_Delay.reset_index(drop=True)
                    df_mean = df_not_day['Dep_Delay'].groupby([df_not_day['Origin_Airport_Code']]).mean().reset_index()
                    m_t[int((day - 1) / 7)] = df_mean["Dep_Delay"]

            dict_values = {}
            # Calculate the Standard Deviation and Z-score
            for j in range(n):
                sum_sq = x_t - (m_t.loc[0,j]) # Second digit is the indexed number
                sum_sq = sum_sq**2
                sum_val = sum_sq.values.sum()
                sd = np.sqrt(sum_val/n)
                z_sc = (x_t.iloc[0,j]-m_t.iloc[0,j])/sd
                prob = st.norm.cdf(z_sc)
                dict_values[j] = [prob]
                df_zscore = pd.DataFrame(dict_values).T

            day_list_series = pd.Series(day_list)
            df_zscore['Day'] = day_list_series.values
            df_zscore.set_index('Day', inplace=True)
            df_zscore_com = pd.concat([df_zscore_com, df_zscore])

        df_zscore_com.sort_index(inplace=True)
        if c == 1:
            df_zscore_all = pd.DataFrame(index=df_zscore_com.index)

        df_zscore_all[d] = df_zscore_com.values
        
    df_zscore_all.insert(loc=0, column = 'Day_of_Year', value=df_zscore_all.index) #Create a column for day of year
    file_combined = "%s%s_%s_Zdata_%s.csv" %(processed_direc,year,airline,include_data)
    df_zscore_all.to_csv(file_combined, index=False)
    print('%s Z-score file created at %s' %(year,file_combined))

def airline_z_score(year,airline,processed_direc,include_data='ADM'):
    """
    Calculate the Z-score for each airline on each day of the year. The value for each day is the average of the 
    average delay of the airports in the airline's filtered network based on the ADM method.

    The Z-scoreis calculated relative to the mean and variance of the other days of week that the
    day of year corresponds to.

    Parameters
    ----------
    db_path: string
        Path to the location of the atn database
    year: int
        Year of data you would like to process.
    airline: string
        Code for the airline to process data for
        Use '' to retrieve for all airlines
    include_data: string
        Specify what kind of data to include in processed flight data. See drop_flights in M-D File. Possible parameters are:
            CC: Cancellations only
            ADD: Arrival delays including diversions
            ADM: Purely arrival delays excluding cancellations or diversions
            DCC: Combined delay. If arrival delay is greater than a set threshold, the flight is considered cancelled
            DD: Departure delays. Does not include cancelled or diverted flights.
    processed_direc: string
        Location of the folder for the processed data csv to be created
        
    Returns
    -------
    file: csv
        Returns a csv in /data/processed with the name year_airline_Zdata_includedata.csv
    
    Notes
    -----
    The atn db must be created first. Please see create_atn_db.
    
    """
    querries = AnalysisQueries()
    #Call the airline_query function below to retrieve the airlines to be used for analysis
    airline_df = querries.airline_query(year,airline)
        
    no_of_days = 366 if calendar.isleap(year) == 1 else 365

    # Build the sets of days of year corresponding to each day of week
    day_list_all = []
    for j in range(1, 8):
        day_list_all.append([])
        for i in range(j, no_of_days + 1, 7):
            day_list_all[-1].append(i)
        
    df_zscore_all = pd.DataFrame(columns=['Day_of_Year','Zscore'])
    for dow in range(7):
        day_list = day_list_all[dow]

        for day_of_year in day_list:
            day_list_exlc_current = copy.deepcopy(day_list)
            day_list_exlc_current.remove(day_of_year)

            today_mean = airline_df.loc[
                airline_df['Day']==day_of_year]['Pos_Arr_Delay'].mean()
            dow_list = airline_df.loc[
                airline_df['Day'].isin(day_list_exlc_current)]
            dow_day_avg = dow_list.groupby('Day').mean()

            dow_mean = dow_day_avg['Pos_Arr_Delay'].mean()
            st_dev = np.std(dow_day_avg['Pos_Arr_Delay'])
            z_sc = (today_mean - dow_mean)/st_dev
            df_zscore_all = df_zscore_all.append(
                {'Day_of_Year':day_of_year,'Zscore':z_sc},ignore_index=True)

    df_zscore_all = df_zscore_all.astype({'Day_of_Year':'int64'})
    df_zscore_all.sort_values(by=['Day_of_Year'],inplace=True)
    df_zscore_all.reset_index(drop=True,inplace=True)

    file_combined = "%s%s_%s_airline_Zdata_%s.csv" %(processed_direc,airline,year,include_data)
    df_zscore_all.to_csv(file_combined, index=False)
    print('%s Z-score file created at %s' %(year,file_combined))
    return(df_zscore_all)

def calculate_airline_zscore_airports(year,airline,processed_direc,include_data='ADM'):
    ''''
    Calculate the airline z-score but return the average for each airport on each day

    '''
    querries = AnalysisQueries()
    #Call the airline_query function below to retrieve the airlines to be used for analysis
    airline_df = querries.airline_query(year,airline)
        
    no_of_days = 366 if calendar.isleap(year) == 1 else 365

    # Build the sets of days of year corresponding to each day of week
    day_list_all = []
    for j in range(1, 8):
        day_list_all.append([])
        for i in range(j, no_of_days + 1, 7):
            day_list_all[-1].append(i)
        
    df_zscore_all = pd.DataFrame(columns=['Day_of_Year','Zscore'])
    for dow in range(7):
        day_list = day_list_all[dow]

        for day_of_year in day_list:
            day_list_exlc_current = copy.deepcopy(day_list)
            day_list_exlc_current.remove(day_of_year)

            today_mean = airline_df.loc[
                airline_df['Day']==day_of_year]['Pos_Arr_Delay'].mean()
            dow_list = airline_df.loc[
                airline_df['Day'].isin(day_list_exlc_current)]
            dow_day_avg = dow_list.groupby('Day').mean()

            dow_mean = dow_day_avg['Pos_Arr_Delay'].mean()
            st_dev = np.std(dow_day_avg['Pos_Arr_Delay'])

def raw_query(db_path,year,airline):
    """
    Queries from the atn_performance db (at specified path) and returns all data based on the year and airline specified
    
    Parameters
    ----------
    db_path: string
        Path to the location of the atn database
    year: int
        Year of data you would like to process.
    airline: string
        Code for the airline to process data for
        Use '' to retrieve for all airlines
    
    Returns
    -------
    df : Dataframe
        A Dataframe with all data for a given year and airline
        
    Notes
    -----
    
    """   
    
    conn = sqlite3.connect(db_path)

    if airline == '' or airline == 'ALL':
        sql = "SELECT * FROM atn_performance WHERE Year = %s" %(year,)
    else:
        sql = "SELECT * FROM atn_performance WHERE Year = %s AND Unique_Carrier_ID = '%s' " %(year,airline)
        
    raw_df = pd.read_sql_query(sql,conn)
    return(raw_df)
    
    
def dropped_query(db_path,year,airline,include_data):
        #Query the sorted, ungrouped data to calculate the total numner of flights.

#include_dict is a dictionary which maps the given include_data to the SQL statement that that parameter corresponds to. Since DCC has an extra condition, it is seperate. 
    include_dict = {"ADD" : "Can_Status != 1 AND Combined_Arr_Delay IS NOT NULL AND", "ADM":"Can_Status != 1 AND Div_Status != 1 AND", "CC":"", "DD":"Can_Status != 1 AND Dep_Delay IS NOT NULL AND", "DCC": "Combined_Arr_Delay IS NOT NULL AND"} 
    #fields that will be imported from the DB
    import_fields = ["Destination_Airport_Code", "Flight_Date", "Day_Of_Year", "Origin_Airport_Code", "Pos_Arr_Delay", "Combined_Arr_Delay", "Can_Status", "Div_Status", "Dep_Delay"]
    
## The following is equivalent to Keshav's original drop_flights script
    
    conn = sqlite3.connect(db_path)
    if airline == '':
        if include_data == "DCC":
            sql = "SELECT %s FROM atn_performance WHERE %s Year = '%s'" %(", ".join(import_fields), include_dict[include_data], year)
            df_int = pd.read_sql_query(sql,conn)
            df_int['Del_Count'] = np.where(df_int['Combined_Arr_Delay'] > 14.99, 1, 0)
        else:
            sql = "SELECT %s FROM atn_performance WHERE %s Year = '%s'" %(", ".join(import_fields), include_dict[include_data], year)
            df_int = pd.read_sql_query(sql,conn)
     
    else:
        if include_data == "DCC":
            sql = "SELECT %s FROM atn_performance WHERE %s Year = %s AND Unique_Carrier_ID = '%s'" %(", ".join(import_fields), include_dict[include_data], year, airline)
            df_int = pd.read_sql_query(sql,conn)
            df_int['Del_Count'] = np.where(df_int['Combined_Arr_Delay'] > 14.99, 1, 0)    
        else:
            sql = "SELECT %s FROM atn_performance WHERE %s Year = %s AND Unique_Carrier_ID = '%s'" %(", ".join(import_fields), include_dict[include_data], year, airline)
            df_int = pd.read_sql_query(sql,conn)
    
    #First determine which airports or routes will be included (depending on the grouping). AKA, which rows we will not drop
    unique_days = df_int.Day_Of_Year.unique() #A list of all days with data
    #Should have a query-like function that can return any relevent origin dates
    day_sets = []
    for day in unique_days:
        df_day = df_int[df_int.Day_Of_Year == day]
        #day_sets is a list of sets. Each sets includes all the airports for that day
        if include_data in ["CC","DD",]:
            day_sets += [set(df_day.Origin_Airport_Code)] 
        else:
            day_sets += [set(df_day.Destination_Airport_Code)] 
    #relevant_airports will return an intersection of the set, in other words, a list of airports that appear in all days for that year. Previously od_set_year in Keshav's code
    
    relevant_airports = set.intersection(*day_sets)
    relevant_airports = sorted(relevant_airports)
    
    #Depending on the sorting used, either drop based on Origin or Destination airport
    if include_data in ["CC","DD",]:
        df_dropped = df_int.reset_index().set_index(['Origin_Airport_Code']).sort_index()
    else:
        df_dropped = df_int.reset_index().set_index(['Destination_Airport_Code']).sort_index()
    
    df_dropped = df_dropped.loc[relevant_airports]
    df_dropped = df_dropped.reset_index().set_index(['index']).sort_index()
    df_dropped.index = range(len(df_dropped.index))
    return(df_dropped)
    
def daily_mean_delays(db_path,year,airline,include_data):
    fds
    
    
def yearly_mean_delays(db_path,years,airlines,include_data,processed_direc):
    #Calculate the total yearly average delays
    #Start counting total number of flights and total delays
    all_metrics = []
    
    for airline in airlines:
        for year in years:
            total_flights = 0
            total_delay = 0
            initial_df = dropped_query(db_path,year,airline,include_data)
            total_flights += len(initial_df)
            total_delay += initial_df['Pos_Arr_Delay'].sum()
            airline_metrics = [year,airline,total_delay,total_flights,total_delay/total_flights]
            all_metrics.append(airline_metrics)
            
    cols = ['Year','Airline','Total Delay','Total Flights','Average Delay']
    metrics_df = pd.DataFrame(all_metrics, columns=cols)
    dropped_path = processed_direc + 'delay_metrics.csv'
    metrics_df.to_csv(dropped_path)
    
    

    
def month_query(db_path,processed_direc):
        
    conn = sqlite3.connect(db_path)
    sql = "SELECT * FROM atn_performance WHERE strftime('%Y-%m',Flight_Date) IN ('2013-03','2013-04')"
    raw_df = pd.read_sql_query(sql,conn)
    
    processed_path = processed_direc+'2013_03_04.csv'
    raw_df.to_csv(processed_path, index=False)
    
def wn_oct18_17_query(db_path,processed_direc):
    conn = sqlite3.connect(db_path)
    sql = "SELECT Origin_Airport_Code, Destination_Airport_Code, Arr_Delay, Pos_Arr_Delay, Flight_Date FROM atn_performance WHERE strftime('%Y-%m-%d',Flight_Date) IN ('2017-08-13') AND Unique_Carrier_ID = 'WN'"
    raw_df = pd.read_sql_query(sql,conn)
    
    processed_path = processed_direc+'wn_8_13_2017.csv'
    raw_df.to_csv(processed_path, index=False)
    
def aa_oct15_2017_query(db_path,processed_direc):
    conn = sqlite3.connect(db_path)
    sql = "SELECT Origin_Airport_Code, Destination_Airport_Code, Arr_Delay, Pos_Arr_Delay, Flight_Date FROM atn_performance WHERE strftime('%Y-%m-%d',Flight_Date) IN ('2017-10-15') AND Unique_Carrier_ID = 'AA'"
    raw_df = pd.read_sql_query(sql,conn)
    
    processed_path = processed_direc+'aa_10_15_2017.csv'
    raw_df.to_csv(processed_path, index=False)

