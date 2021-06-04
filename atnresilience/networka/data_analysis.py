import os
import pandas as pd
import numpy as np


def filter_fields(file_name):
    """ Filters raw data from existing BTS csv files and creates a processed
     file with only required fields

        Args :
        file_name: Takes a string of the file name as an argument

        Returns:
            'Processed file as a csv file in the sub-folder "processed"'
            filtered_df.T: Array of filtered data.
    """

    # Path for source (raw data, Any folder -> data -> raw)
    script_dir = os.path.dirname(os.getcwd())
    rel_path = "data/raw/%s" % file_name
    raw_file = os.path.join(script_dir, rel_path)
    # dtypes = [int64,int64,int64,int64,int64,object,object,int64,object,object,
    #           int64,int64,int64,int64,object,object,object,int64,object,int64,
    #           int64,int64,int64,object,object,object,int64,object,int64,int64,
    #           float64,float64,float64,float64,float64,object,float64,float64,
    #           float64,float64,int64,float64,float64,float64,float64,float64,
    #           object,float64,object,float64,float64,float64,float64,float64,
    #           float64,int64,float64,float64,float64,float64,float64,float64,
    #           float64,float64,int64,float64,float64,float64,float64,object,
    #           float64,float64,float64,float64,float64,float64,object,object,
    #           float64,float64,float64,float64,float64,float64,object,object,
    #           float64,float64,float64,float64,float64,float64,object,object,
    #           float64,float64,float64,float64,float64,float64,object,object,
    #           float64,float64,float64,float64,float64,float64,object]

    # df = pd.read_csv(raw_file, parse_dates=['FL_DATE'])
    df = pd.read_csv(raw_file)
    df.loc[df.CRS_DEP_TIME == 2400, 'CRS_DEP_TIME'] = 0000  # Convert 2400 hrs to 0000 hrs
    # df = df[np.isfinite(df['CRS_ARR_TIME'])]
    df.loc[df.CRS_ARR_TIME == 2400, 'CRS_ARR_TIME'] = 0000
    df.CRS_DEP_TIME = df.CRS_DEP_TIME.astype(str).str.zfill(4)  # Filling 0s for times that only have 2 digits
    df.DEP_TIME = df.DEP_TIME.astype(str).str.zfill(4)
    df.CRS_ARR_TIME = df.CRS_ARR_TIME.astype(str).str.zfill(4)
    df.ARR_TIME = df.ARR_TIME.astype(str).str.zfill(4)
    df.CRS_DEP_TIME = pd.to_datetime(df.CRS_DEP_TIME, format='%H%M').dt.time
    df.CRS_ARR_TIME = pd.to_datetime(df.CRS_ARR_TIME, format='%H%M').dt.time
    df.FL_DATE_TIME = df.FL_DATE.astype(str) + ' ' + df.CRS_DEP_TIME.astype(str)  # Combining flight date and time
    df.DAY_OF_YEAR = pd.to_datetime(df.FL_DATE_TIME).dt.dayofyear
    # Take normal arrival delay for regular flights or diversion delay in the case of diverted flights
    df['COMBINED_ARR_DELAY'] = df[['ARR_DELAY', 'DIV_ARR_DELAY']].max(axis=1)
    # df.DIV_DELAY = df.ARR_DELAY.fillna(0) + df.DIV_ARR_DELAY.fillna(0)

    filtered_data = [df.FL_DATE, df.FL_DATE_TIME, df.DAY_OF_YEAR, df.UNIQUE_CARRIER, df.AIRLINE_ID, df.TAIL_NUM,
                     df.FL_NUM, df.ORIGIN_AIRPORT_ID, df.ORIGIN_CITY_MARKET_ID, df.ORIGIN, df.ORIGIN_STATE_ABR,
                     df.DEST_AIRPORT_ID, df.DEST_CITY_MARKET_ID, df.DEST, df.DEST_STATE_ABR,
                     df.CRS_DEP_TIME, df.DEP_TIME, df.DEP_DELAY, df.DEP_DELAY_NEW,
                     df.CRS_ARR_TIME, df.ARR_TIME, df.ARR_DELAY, df.ARR_DELAY_NEW, df.COMBINED_ARR_DELAY,
                     df.CANCELLED, df.CANCELLATION_CODE, df.DIVERTED, df.CRS_ELAPSED_TIME,
                     df.ACTUAL_ELAPSED_TIME,
                     df.CARRIER_DELAY, df.WEATHER_DELAY, df.NAS_DELAY, df.SECURITY_DELAY, df.LATE_AIRCRAFT_DELAY,
                     df.DIV_AIRPORT_LANDINGS, df.DIV_REACHED_DEST, df.DIV_ACTUAL_ELAPSED_TIME, df.DIV_ARR_DELAY,
                     df.DIV1_AIRPORT_ID, df.DIV1_TAIL_NUM, df.DIV2_AIRPORT_ID, df.DIV2_TAIL_NUM,
                     df.DIV3_AIRPORT_ID, df.DIV3_TAIL_NUM, df.DIV4_AIRPORT_ID, df.DIV4_TAIL_NUM,
                     df.DIV5_AIRPORT_ID, df.DIV5_TAIL_NUM]
    header_names = ["Flight_Date", "Date_Time", "Day_Of_Year", "Unique_Carrier_ID", "Airline_ID", "Tail_Number",
                    "Flight_Number", "Origin_Airport_ID", "Origin_Market_ID ", "Origin_Airport_Code", "Origin_State",
                    "Destination_Airport_ID", "Destination_Market_ID ", "Destination_Airport_Code", "Dest_State",
                    "Scheduled_Dep_Time", "Actual_Dep_Time", "Dep_Delay", "Pos_Dep_Delay",
                    "Scheduled_Arr_Time", "Actual_Arr_Time", "Arr_Delay", "Pos_Arr_Delay", "Combined_Arr_Delay",
                    "Can_Status", "Can_Reason", "Div_Status", "Scheduled_Elapsed_Time",
                    "Actual_Elapsed_Time", "Carrier_Delay", "Weather_Delay", "Natl_Airspace_System_Delay",
                    "Security_Delay", "Late_aircraft_delay", "Div_airport_landings", "Div_Landing_Status",
                    "Div_Elapsed_Time", "Div_Arrival_Delay", "Div_Airport_1_ID", "Div_1_Tail_#",
                    "Div_Airport_2_ID", "Div_2_Tail_#", "Div_Airport_3_ID", "Div_3_Tail_#",
                    "Div_Airport_4_ID", "Div_4_Tail_#", "Div Airport_5_ID", "Div_5_Tail_#"]

    file_processed = file_name.split('.', 1)[0]
    rel_path_2 = "data/processed/%s_processed.csv" % file_processed
    processed_file = os.path.join(script_dir, rel_path_2)
    filtered_df = pd.DataFrame(filtered_data)
    filtered_df.T.to_csv(processed_file, index=False, header=header_names)
    return filtered_df.T
