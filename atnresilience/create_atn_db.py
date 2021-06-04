import os
from datetime import datetime as datetime
from sqlalchemy import create_engine
import sqlite3

import pandas as pd

# print(os.getcwd())
import db_tools as DBTools
# if __name__ == "__main__":
#     import db_tools as DBTools

# else:
#     import atnresilience.db_tools as DBTools


class ATNLoader(DBTools.DBLoader):
    def __init__(self, db_path = None, raw_path = None):
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

    def col_parse(self, cols):
        """
        Creates the columns string which will be used to create the db, 
        since we are taking a list of columns, the list must be converted
        to a string including the data type to be used as a SQL command.
        The function has all possible column names and will give them the 
        approriate type. If name is given that is not a column in the data, 
        it will return "Columns does not exist in data".
    
        Parameters
        ----------
        cols : list
            Specify the column names to import from the raw data.
            
        Returns
        -------
        db_cols : string
            The columns with type in the db.
        
        Notes
        -----
        Since SQL command to take the columns is a string that requires data types and the pandas command
        (used in insert_data function) to import the db data is a list, the cols list needs to be converted
        to the SQL command which is handled by the col_parse function
        
        """
        
        db_cols = ""
        
        int_not_null = ['Quarter','Month','Day_of_Month','Day_Of_Week','Airline_ID','Origin_Airport_ID','Origin_Airport_Seq_ID','Origin_Market_ID','OriginWac','Destination_Airport_ID','DestAirportSeqID','Destination_Market_ID','Can_Status']

        int_null = ['Year','Flight_Number','Day_Of_Year','Origin_State_Fips','Destination_State_Fips','Dest_Wac','Dep_Delay','Pos_Dep_Delay','Dep_Del_15','Departure_Delay_Groups','Taxi_Out','Taxi_In','Arr_Delay','Pos_Arr_Delay','Arr_Del_15','Arrival_Delay_Minutes','Arr_Del_15','Arrival_Delay_Groups','Div_Status','Scheduled_Elapsed_Time','Actual_Elapsed_Time','Air_Time','Flights','Distance','Distance_Group','Carrier_Delay','Weather_Delay','Natl_Airspace_System_Delay','Security_Delay','Late_Aircraft_Delay','Total_Add_G_Time','Longest_Add_G_Time','Div_Airport_Landings','Div_Landing_Status','Div_Elapsed_Time','Div_Arrival_Delay','Div_Distance','Div_Airport_1_ID','Div_Airport_1_Seq_ID','Div_1_Total_G_Time','Div_1_Longest_G_Time','Div_Airport_2_ID','Div_Airport_2_ID','Div_2_Total_G_Time','Div_2_Longest_G_Time','Div_Airport_3_ID','Div_Airport_3_Seq_ID','Div_3_Total_G_Time','Div_3_Longest_G_Time','Div_Airport_4_ID','Div_Airport_4_Seq_ID','Div_4_Total_G_Time','Div_4_Longest_G_Time','Div_Airport_5_ID','Div_Airport_5_Seq_ID','Div_5_Total_G_Time','Div_5_Longest_G_Time','Combined_Arr_Delay']

        date_not_null = ['Flight_Date']

        time_not_null = ['Scheduled_Dep_Time','Scheduled_Arr_Time']

        var_10_null = ['Unique_Carrier_ID','Carrier','Origin_Airport_Code','Origin_State','Destination_Airport_Code','Actual_Dep_Time']

        var_10_not_null = ['Dest_State']

        var_45_null = ['Tail_Number','Origin_City_Name','Origin_State_Name','Dest_City_Name','Dest_State_Name','Dep_Time_Blk','Wheels_Off','Wheels_On','Actual_Arr_Time','Arr_Time_Blk','Can_Reason','First_Dep_Time','Div_Airport_1','Div_1_Wheels_On','Div_1_Wheels_Off','Div_1_Tail_Num','Div_Airport_2','Div_2_Wheels_On','Div_2_Wheels_Off','Div_2_Tail_Num','Div_Airport_3','Div_3_Wheels_On','Div_3_Wheels_Off','Div_3_Tail_Num','Div_Airport_4','Div_4_Wheels_On','Div_4_Wheels_Off','Div_4_Tail_Num','Div_Airport_5','Div_5_Wheels_On','Div_5_Wheels_Off','Div_5_Tail_Num']

        #Read the provided cols list and create the string for the db columns while appending the data types. If a column name is given but it not a column given in the raw data, it will print does not exist.
        for line in cols:
            if line in int_not_null:
                db_cols = db_cols + ",'" + line + "' INT NULL"
            elif line in int_null:
                db_cols = db_cols + ",'" + line + "' INT"
            elif line in date_not_null:
                db_cols = db_cols + ",'" + line + "' DATE NULL"
            elif line in time_not_null:
                db_cols = db_cols + ",'" + line + "' TIME NULL"
            elif line in var_10_null:
                db_cols = db_cols + ",'" + line + "' VARCHAR(10) NULL"
            elif line in var_10_not_null:
                db_cols = db_cols + ",'" + line + "' VARCHAR(10) NULL"
            elif line in var_45_null:
                db_cols = db_cols + ",'" + line + "' VARCHAR(45)"
            else:
                print("Column %s does not exist in data." %line)
        #Return 1: because the first item will be a comma which we will want to omit
        return(db_cols[1:])

    def create_db(self):
        """
        Creates the table atn_performance in the database at the specified input location if one does not exist.

        Parameters
        ----------
        db_path: string
            Path to the location of the atn database
        cols: list
            A list of column titles that will be created on the SQL table
        
        Returns
        -------
        Creates a db at the give path. 
        If one already exists, no action will be taken.
        
        Notes
        -----
        
        """
        
        #Specify columns for the database table based on data used. Does not need to be changed
        
        cols = ["Year", "Flight_Date", "Day_Of_Year", "Unique_Carrier_ID", "Airline_ID", "Tail_Number", "Flight_Number", 
            "Origin_Airport_ID", "Origin_Market_ID", "Origin_Airport_Code", "Origin_State", "Destination_Airport_ID", 
            "Destination_Market_ID", "Destination_Airport_Code", "Dest_State", "Scheduled_Dep_Time", "Actual_Dep_Time", 
            "Dep_Delay", "Pos_Dep_Delay", "Scheduled_Arr_Time", "Actual_Arr_Time", "Arr_Delay", "Pos_Arr_Delay", 
            "Combined_Arr_Delay", "Can_Status", "Can_Reason", "Div_Status", "Scheduled_Elapsed_Time", 
            "Actual_Elapsed_Time", "Carrier_Delay", "Weather_Delay", "Natl_Airspace_System_Delay", "Security_Delay", 
            "Late_Aircraft_Delay", "Div_Airport_Landings", "Div_Landing_Status", "Div_Elapsed_Time", "Div_Arrival_Delay", 
            "Div_Airport_1_ID", "Div_1_Tail_Num", "Div_Airport_2_ID", "Div_2_Tail_Num", "Div_Airport_3_ID", "Div_3_Tail_Num", 
            "Div_Airport_4_ID", "Div_4_Tail_Num", "Div_Airport_5_ID", "Div_5_Tail_Num"]

        db_cols = self.col_parse(cols)

        sql = '''
            CREATE TABLE IF NOT EXISTS atn_performance (
                %s,
                UNIQUE(Flight_Date, Origin_Airport_ID, Unique_Carrier_ID, Flight_Number) ON CONFLICT REPLACE
                )
            '''%(db_cols,)
        
        # Execute the SQL statement
        self.db_query(sql)
        

    def import_data(self, year):
        """
        Imports the data for a specified year (all 12 months) to the database.
        Fixed columns are imported. This is based on the project spec.
        
        Parameters
        ----------
        db_path: string
            Path to the location of the atn database
        raw_path: string
            Path to the location of the raw data folder
        year: int
            Year of data to import
        
        Returns
        -------
        Does not return any specific data, but will return prompts when finished running.

        
        Notes
        -----
        
        """
            
        #Take the raw csv file for each month, process it based on parameters, and append it to the dataframe to be exported to the db.

        import_cols = ['FL_DATE', 'UNIQUE_CARRIER', 'OP_UNIQUE_CARRIER', 'AIRLINE_ID', 'OP_CARRIER_AIRLINE_ID', 
            'TAIL_NUM', 'FL_NUM', 'OP_CARRIER_FL_NUM', 'ORIGIN_AIRPORT_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 
            'ORIGIN_STATE_ABR', 'DEST_AIRPORT_ID', 'DEST_CITY_MARKET_ID', 'DEST', 'DEST_STATE_ABR', 'CRS_DEP_TIME', 
            'DEP_TIME', 'DEP_DELAY', 'DEP_DELAY_NEW', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'ARR_DELAY_NEW', 'CANCELLED', 
            'CANCELLATION_CODE', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'CARRIER_DELAY', 'WEATHER_DELAY', 
            'NAS_DELAY','SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'DIV_AIRPORT_LANDINGS', 'DIV_REACHED_DEST', 'DIV_ACTUAL_ELAPSED_TIME', 
            'DIV_ARR_DELAY', 'DIV1_AIRPORT_ID', 'DIV1_TAIL_NUM', 'DIV2_AIRPORT_ID', 'DIV2_TAIL_NUM', 'DIV3_AIRPORT_ID', 'DIV3_TAIL_NUM', 
            'DIV4_AIRPORT_ID', 'DIV4_TAIL_NUM', 'DIV5_AIRPORT_ID', 'DIV5_TAIL_NUM']        
        
        import_dict = {'FL_DATE': 'Flight_Date', 
            'UNIQUE_CARRIER': 'Unique_Carrier_ID', 
            'OP_UNIQUE_CARRIER': 'Unique_Carrier_ID', 
            'AIRLINE_ID': 'Airline_ID',
            'OP_CARRIER_AIRLINE_ID': 'Airline_ID', 
            'TAIL_NUM': 'Tail_Number', 
            'FL_NUM': 'Flight_Number', 
            'OP_CARRIER_FL_NUM': 'Flight_Number', 
            'ORIGIN_AIRPORT_ID': 'Origin_Airport_ID', 
            'ORIGIN_CITY_MARKET_ID': 'Origin_Market_ID', 
            'ORIGIN': 'Origin_Airport_Code', 
            'ORIGIN_STATE_ABR': 'Origin_State', 
            'DEST_AIRPORT_ID': 'Destination_Airport_ID', 
            'DEST_CITY_MARKET_ID': 'Destination_Market_ID', 
            'DEST': 'Destination_Airport_Code', 
            'DEST_STATE_ABR': 'Dest_State', 
            'CRS_DEP_TIME': 'Scheduled_Dep_Time', 
            'DEP_TIME': 'Actual_Dep_Time', 
            'DEP_DELAY': 'Dep_Delay', 
            'DEP_DELAY_NEW': 'Pos_Dep_Delay', 
            'CRS_ARR_TIME': 'Scheduled_Arr_Time', 
            'ARR_TIME': 'Actual_Arr_Time', 
            'ARR_DELAY': 'Arr_Delay', 
            'ARR_DELAY_NEW': 'Pos_Arr_Delay', 
            'CANCELLED': 'Can_Status', 
            'CANCELLATION_CODE': 'Can_Reason', 
            'DIVERTED': 'Div_Status', 
            'CRS_ELAPSED_TIME': 'Scheduled_Elapsed_Time', 
            'ACTUAL_ELAPSED_TIME': 'Actual_Elapsed_Time', 
            'CARRIER_DELAY': 'Carrier_Delay', 
            'WEATHER_DELAY': 'Weather_Delay', 
            'NAS_DELAY': 'Natl_Airspace_System_Delay', 
            'SECURITY_DELAY': 'Security_Delay', 
            'LATE_AIRCRAFT_DELAY': 'Late_Aircraft_Delay', 
            'DIV_AIRPORT_LANDINGS': 'Div_Airport_Landings', 
            'DIV_REACHED_DEST': 'Div_Landing_Status', 
            'DIV_ACTUAL_ELAPSED_TIME': 'Div_Elapsed_Time', 
            'DIV_ARR_DELAY': 'Div_Arrival_Delay', 
            'DIV1_AIRPORT_ID': 'Div_Airport_1_ID',
            'DIV1_TAIL_NUM': 'Div_1_Tail_Num', 
            'DIV2_AIRPORT_ID': 'Div_Airport_2_ID', 
            'DIV2_TAIL_NUM': 'Div_2_Tail_Num', 
            'DIV3_AIRPORT_ID': 'Div_Airport_3_ID', 
            'DIV3_TAIL_NUM': 'Div_3_Tail_Num', 
            'DIV4_AIRPORT_ID': 'Div_Airport_4_ID', 
            'DIV4_TAIL_NUM': 'Div_4_Tail_Num', 
            'DIV5_AIRPORT_ID': 'Div_Airport_5_ID', 
            'DIV5_TAIL_NUM': 'Div_5_Tail_Num'}    
        #Loop through the csv for each month of the year.
        #During each iteration, create the Combined_Arr_Delay column and make sure that the CRS Dep and ARR columns follow the 0000 format. In each loop, the data from the month will be added to the db
        for i in range(1,13):
            csv_import_cols = []
            month = str(year*100+i)
            csv_cols = list(pd.read_csv('%s%s.csv' %(self.raw_path,month)).columns.values)
            #For every column header in the csv, check if it is a column to be imported (import_col)
            for column in csv_cols:
                if column in import_cols:
                    csv_import_cols.append(column)
                else:
                    continue
            raw_file = '%s%s.csv' %(self.raw_path,month)
            import_df = pd.read_csv(raw_file, low_memory=False, usecols = csv_import_cols)
            import_df['Combined_Arr_Delay'] = import_df[['ARR_DELAY', 'DIV_ARR_DELAY']].max(axis=1)
            
            if "CRS_DEP_TIME" in import_cols:
                import_df.loc[import_df.CRS_DEP_TIME == 2400, 'CRS_DEP_TIME'] = 0000
                import_df.CRS_DEP_TIME = import_df.CRS_DEP_TIME.astype(str).str.zfill(4)
            if "CRS_ARR_TIME" in import_cols:
                import_df.loc[import_df.CRS_ARR_TIME == 2400, 'CRS_ARR_TIME'] = 0000
                import_df.CRS_ARR_TIME = import_df.CRS_ARR_TIME.astype(str).str.zfill(4)
            #df_all = pd.concat([df_all,import_df],ignore_index=True)
            #df_all.append(import_df, ignore_index=True)
            
            import_df.rename(columns=import_dict, inplace=True) #change all the col names
            
            import_df['Day_Of_Year'] = pd.to_datetime(import_df.Flight_Date.astype(str) + ' ' + import_df.Scheduled_Dep_Time.astype(str)).dt.dayofyear
            
            import_df['Year'] = year
            
            # Load to database
            self.df_to_db('atn_performance', import_df)

            print("Finished inserting month %s to DB." %(i,))
            
        print("Finished inserting data for year %s" %(year,))
            
        #After the df_all dataframe is created fro the whole year, change the column names based on rename_cols

        #renamed_cols = ["Flight_Date", "Date_Time", "Unique_Carrier_ID", "Airline_ID", "Tail_Number", "Flight_Number", "Origin_Airport_ID", "Origin_Market_ID ", "Origin_Airport_Code", "Origin_State", "Destination_Airport_ID", "Destination_Market_ID", "Destination_Airport_Code", "Dest_State", "Scheduled_Dep_Time", "Actual_Dep_Time", "Dep_Delay", "Pos_Dep_Delay", "Scheduled_Arr_Time", "Actual_Arr_Time", "Arr_Delay", "Pos_Arr_Delay", "Can_Status", "Can_Reason", "Div_Status", "Scheduled_Elapsed_Time", "Actual_Elapsed_Time", "Carrier_Delay", "Weather_Delay", "Natl_Airspace_System_Delay","Security_Delay", "Late_Aircraft_Delay", "Div_Airport_Landings", "Div_Landing_Status", "Div_Elapsed_Time", "Div_Arrival_Delay", "Div_Airport_1_ID", "Div_1_Tail_#", "Div_Airport_2_ID", "Div_2_Tail_#", "Div_Airport_3_ID", "Div_3_Tail_#", "Div_Airport_4_ID", "Div_4_Tail_#", "Div Airport_5_ID", "Div_5_Tail_#"]

class CoordinateLoader(DBTools.DBLoader):
    def __init__(self, db_path = None, raw_path = None):
        """
        Initialize common parameters used for the coordinate database and inputs.
        If none given, they are defaulted to their expected locations within the repository.
        Default Database Location: /data/processed/atn_db.sqlite
        Default Raw Data Folder: /data/raw/

        Parameters
        ----------
        db_path: string
            Path to the database
        
        Returns
        ----------
        None 
            Performs the query passed

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
    
    def create_coords_table(self):
        """
        Creates the table airportCoords in the database at the specified input location if one does not exist.
        Data can be downloaded from openflights: https://openflights.org/data.html
        The extended dataset is used as some airports that appear in BTS data is not in the "clean" set. 

        Parameters
        ----------
        None
        
        Returns
        -------
        None
            Creates a db at the give path. 
            If one already exists, no action will be taken.
        
        """
        
        ##Change column names to match the ones from csv

        query = '''
        CREATE TABLE IF NOT EXISTS airportCoords(
            IATA TEXT,
            lat DECIMAL,
            long DECIMAL,
            UNIQUE(IATA) ON CONFLICT REPLACE)

        '''

        self.db_query(query)
        
    def import_coords_data(self):
        """
        Imports the airport coordinate data to the database.
        
        Parameters
        ----------
        db_path: string
            Path to the location of the atn database
        raw_path: string
            Path to the location of the raw data folder
        year: int
            Year of data to import
        
        Returns
        -------
        Does not return any specific data, but will return prompts when finished running.

        
        Notes
        -----
        
        """
        import_cols = ['IATA', 'lat', 'long']
        
        load_file_path = os.path.join(self.raw_path,'airport_data.csv')
        
        coord_df = pd.read_csv(load_file_path, usecols = import_cols)
        
        self.df_to_db('airportCoords', coord_df)
    
class ACDataLoader(DBTools.DBLoader):
    """
    Create and load a database for US aircraft registration data from the FAA N-Number registry database.
    """
    def __init__(self, db_path = None, raw_path = None):
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

    def create_registry_table(self):
        """
        Creates the table nnum_master in the database at the specified input location if one does not exist.

        The nnum_mater table contains the data from MASTER from the FAA N-Number registry that provides
        data on each releaseable aircraft registered with United States FAA and associated airframe and engine 
        parameters such as the aircraft type, owner, airworthiness, productino year, and location of registration.


        The FAA data can be obtained from:
        https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry/releasable_aircraft_download/

        Parameters
        ----------
        None
        
        Returns
        -------
        None
            Creates a db at the give path. 
            If one already exists, no action will be taken.
        
        """
        query = '''
            CREATE TABLE IF NOT EXISTS nnum_master(
            N_NUMBER TEXT, 
            NAME TEXT,
            MFR_CODE TEXT,
            SERIAL_NUM TEXT,
            ISSUE_DATE TEXT,
            UNIQUE(N_NUMBER) ON CONFLICT REPLACE
            )
        '''

        self.db_query(query)

    def create_dereg_table(self):
        """
        Creates the table nnum_dereg in the database at the specified input location if one does not exist.

        The nnum_mater table contains the data from MASTER from the FAA N-Number registry that provides
        data on each releaseable aircraft registered with United States FAA and associated airframe and engine 
        parameters such as the aircraft type, owner, airworthiness, productino year, and location of registration.


        The FAA data can be obtained from:
        https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry/releasable_aircraft_download/

        Parameters
        ----------
        None
        
        Returns
        -------
        None
            Creates a db at the give path. 
            If one already exists, no action will be taken.
        
        """
        query = '''
            CREATE TABLE IF NOT EXISTS nnum_dereg(
            N_NUMBER TEXT, 
            NAME TEXT,
            MFR_CODE TEXT,
            SERIAL_NUM TEXT,
            ISSUE_DATE TEXT,
            CANCEL_DATE TEXT,
            UNIQUE(N_NUMBER) ON CONFLICT REPLACE
            )
        '''

        self.db_query(query)

    def create_ac_ref_table(self):
        '''
        Creates the table ac_ref in the specified input location if one does not exist.

        The ac_ref table contains the data from ACTREF from the FAA N-Number registry that provides
        data on each aircraft type such as the manufacturer, speed, and number of seats.
        '''
        query = '''
            CREATE TABLE IF NOT EXISTS ac_ref(
            CODE TEXT, 
            NO_SEATS INTEGER,
            MFR TEXT,
            MODEL TEXT,
            AC_WEIGHT TEXT,
            SPEED INT,
            TYPE_AC INT,
            UNIQUE(CODE) ON CONFLICT REPLACE
            )
        '''

        self.db_query(query)

    def import_current_nnum_data(self):
        """
        Imports the FAA n-number registry and ACTREF data into the database tables.
        The files MASTER.txt and ACTREF.txt must be in the data/raw/ folder.

        If the issue date in the MASTER is empty, it will be filled with 1990-01-01.
        
        """
        nnum_col_dict = {
            'N-Number'          :   'N_NUMBER',
            'MFR MDL Code'      :   'MFR_CODE',
            'Serial Number'     :   'SERIAL_NUM',
            'Cert Issue Date'   :   'ISSUE_DATE'
        }
        nnum_import_cols = ['N-Number', 'MFR MDL Code', 'Serial Number','Cert Issue Date']
        
        nnum_load_file_path = os.path.join(self.raw_path,'MASTER.txt')

        nnum_df = pd.read_csv(nnum_load_file_path,usecols=nnum_import_cols)#
        nnum_df.rename(columns=nnum_col_dict,inplace=True)
        
        # Cleanup and prepend N to N number
        nnum_df['N_NUMBER'] = 'N' + nnum_df['N_NUMBER'].apply(lambda x: x.strip())
        
        # fill empty issue dates with Jan 1 1990
        # nnum_df['ISSUE_DATE'] = nnum_df['ISSUE_DATE'].apply(lambda x: '19900101' if x.strip() == '' else x)

        nnum_df['ISSUE_DATE'] = nnum_df['ISSUE_DATE'].apply(
            lambda x: '1990-01-01' if x.strip() == '' else datetime.strptime(x,'%Y%m%d').strftime('%Y-%m-%d'))
        
        # nnum_df['ISSUE_DATE'] = nnum_df['ISSUE_DATE'].apply(
        #     lambda x: datetime.strptime(x,'%Y%m%d').strftime('%Y-%m-%d'))

        self.df_to_db('nnum_master',nnum_df)
    
    def import_dereg_nnum_data(self):
        """
        Imports the FAA deregistered n-number registry into the database tables.
        The file DEREG.txt must be in the data/raw/ folder.

        If the issue date in the DEREG file is empty, it will be filled with 1900-01-01.

        If the cancel date in the DEREG file is empty, it will be filled with 1990-01-01.
        
        """
        nnum_col_dict = {
            'N-NUMBER'          :   'N_NUMBER',
            'MFR-MDL-CODE'      :   'MFR_CODE',
            'SERIAL-NUMBER'     :   'SERIAL_NUM',
            'CERT-ISSUE-DATE'   :   'ISSUE_DATE',
            'CANCEL-DATE'       :   'CANCEL_DATE'
        }
        nnum_import_cols = ['N-NUMBER', 'MFR-MDL-CODE', 'SERIAL-NUMBER','CERT-ISSUE-DATE','CANCEL-DATE']
        
        nnum_load_file_path = os.path.join(self.raw_path,'DEREG.txt')

        nnum_df = pd.read_csv(nnum_load_file_path,usecols=nnum_import_cols)
        nnum_df.rename(columns=nnum_col_dict,inplace=True)
        
        # Cleanup
        nnum_df['N_NUMBER'] = 'N' + nnum_df['N_NUMBER'].apply(lambda x: x.strip())
        # Fill empty cells or convert to datetime format otherwise
        nnum_df['ISSUE_DATE'] = nnum_df['ISSUE_DATE'].apply(
            lambda x: '1900-01-01' if x.strip() == '' else datetime.strptime(x,'%Y%m%d').strftime('%Y-%m-%d'))
        
        nnum_df['CANCEL_DATE'] = nnum_df['CANCEL_DATE'].apply(
            lambda x: '1990-01-01' if x.strip() == '' else datetime.strptime(x,'%Y%m%d').strftime('%Y-%m-%d'))

        self.df_to_db('nnum_dereg',nnum_df)
    
    def import_acref_data(self):
        """
        Imports the FAA ACTREF data into the database tables.
        The file ACTREF.txt must be in the data/raw/ folder.
        
        """
        acref_col_dict = {
            'CODE'          :   'CODE',
            'NO-SEATS'      :   'NO_SEATS',
            'MFR'           :   'MFR',
            'MODEL'         :   'MODEL',
            'AC-WEIGHT'     :   'AC_WEIGHT',
            'SPEED'         :   'SPEED',
            'TYPE-ACFT'     :   'TYPE_AC'
        }
        acref_import_cols = ['CODE', 'NO-SEATS', 'MFR', 'MODEL', 'AC-WEIGHT','SPEED','TYPE-ACFT']
        
        acref_load_file_path = os.path.join(self.raw_path,'AcftRef.txt')
        
        self.csv_loader(acref_load_file_path,acref_import_cols,acref_col_dict,'ac_ref')

def main():
    pass

if __name__ == "__main__":
    main()