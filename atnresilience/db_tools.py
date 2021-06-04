import os
import sqlite3
import pandas as pd

class DBLoader(object):
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

    def db_query(self, query):
        """
        Connect to the database, execute a query, and close the database.
        
        Parameters
        ----------
        query: string
            A SQL query to execute
        
        Returns
        ----------
        None 
            Performs the query passed
        """
        # print(self.db_path)

        conn = sqlite3.connect(self.db_path)

        conn.execute(query)

        conn.commit()
        conn.close()

    def df_to_db(self,table_name,df):
        """
        Connect to a dabase and append a pandas dataframe to it
        
        Parameters
        ----------
        df: DataFrame Object

        Returns
        ----------
        None
            Appends the passed dataframe to the database
        """

        conn = sqlite3.connect(self.db_path)

        pd.DataFrame.to_sql(self=df, name = table_name, con=conn, if_exists='append', index=False,chunksize=100000)

        conn.close()

    def csv_loader(self,load_file_path,import_cols,col_dict,table_name):
        """
        Load a comma delimited file into a Database table
        
        Parameters:
        -----------
        import_cols: list
            A list of columns to import from the file
        load_file_path: string
            Path to the file to open
        col_dict: dictionary
            Dictionary of column names that need to be remapped before appending to the db table
        table_name: string
            Name of the table to load the data to
        """
        
        load_df = pd.read_csv(load_file_path, usecols = import_cols)

        load_df.rename(columns=col_dict, inplace=True)
        
        self.df_to_db(table_name, load_df)

class DBQueries(object):
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

    def query_to_df(self,query,params=None):
        '''
        Executes a query and returns the data as a pandas dataframe

        Parameters:
        ----------
        query: string
            SQL query to execute
        params: string or list-like
            Passes parameters to panda's to_sql params used in
            SQL parameterization

        Returns:
        ----------
        Pandas DataFrame of the query results
        '''
        
        conn = sqlite3.connect(self.db_path)
        db_df = pd.read_sql(query,conn,params=params)
        conn.close()

        return(db_df)

    def query_timeframe(self,start_date,end_date):
        '''
        Retrieve all data in the database within a specified time frame

        Parameters:
        ----------
        start_date: string
            In the format of 'YYYY-MM-DD'
        end_date: string
            In the format of 'YYYY-MM-DD'
            
        Returns:
        ----------
        Pandas DataFrame of the query results
        '''
        sql = '''
            SELECT * FROM atn_performance 
            WHERE Flight_Date > :start
            AND Flight_Date < :end
            '''
        
        df = pd.read_sql(sql,conn,params = {'start':start_date,'end':end_date})
        conn.close()

        return(df)