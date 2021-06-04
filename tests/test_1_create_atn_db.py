from atnresilience import create_atn_db as create
import sqlite3
import pandas as pd

raw_path = 'tests/raw/'
db_path = 'tests/processed/test_db.sqlite'

def test_import_data():
    create.create_db(db_path)
    create.import_data(db_path,raw_path, 2015)

    conn = sqlite3.connect(db_path)    
    sql = 'SELECT * FROM atn_performance'
    df_test = pd.read_sql_query(sql,conn)
    
    assert df_test['Year'][0] == 2015 and \
    df_test['Flight_Date'][0] == '2015-01-14' and \
    df_test['Day_Of_Year'][0] == 14 and \
    df_test['Unique_Carrier_ID'][0] == 'WN' and \
    df_test['Tail_Number'][0] == 'N7811F' and \
    df_test['Flight_Number'][0] == 114 and \
    df_test['Origin_Airport_Code'][0] == 'PHL' and \
    df_test['Origin_State'][0] == 'PA' and \
    df_test['Destination_Airport_Code'][0] == 'ATL' and \
    df_test['Dest_State'][0] == 'GA' and \
    df_test['Scheduled_Dep_Time'][0] == 605 and \
    df_test['Actual_Dep_Time'][0] == '600.0' and \
    df_test['Dep_Delay'][0] == -5 and \
    df_test['Pos_Dep_Delay'][0] == 0 and \
    df_test['Scheduled_Arr_Time'][0] == 835 and \
    df_test['Actual_Arr_Time'][0] == '813.0' and \
    df_test['Arr_Delay'][0] == -22 and \
    df_test['Pos_Arr_Delay'][0] == 0 and \
    df_test['Combined_Arr_Delay'][0] == -22 and \
    df_test['Can_Status'][0] == 0 and \
    df_test['Scheduled_Elapsed_Time'][0] == 150 and \
    df_test['Actual_Elapsed_Time'][0] == 133        
    
#    assert df_test['Year'][0] == 2015 and \
#    df_test['Flight_Date'][0] == '2015-01-24' and \
#    df_test['Day_Of_Year'][0] == 24 and \
#    df_test['Unique_Carrier_ID'][0] == 'WN' and \
#    df_test['Tail_Number'][0] == 'N7732A' and \
#    df_test['Flight_Number'][0] == 3205 and \
#    df_test['Origin_Airport_Code'][0] == 'DCA' and \
#    df_test['Origin_State'][0] == 'VA' and \
#    df_test['Destination_Airport_Code'][0] == 'BNA' and \
#    df_test['Dest_State'][0] == 'TN' and \
#    df_test['Scheduled_Dep_Time'][0] == 1425 and \
#    df_test['Actual_Dep_Time'][0] == '1424.0' and \
#    df_test['Dep_Delay'][0] == -1 and \
#    df_test['Pos_Dep_Delay'][0] == 0 and \
#    df_test['Scheduled_Arr_Time'][0] == 1525 and \
#    df_test['Actual_Arr_Time'][0] == '1501.0' and \
#    df_test['Arr_Delay'][0] == -24 and \
#    df_test['Pos_Arr_Delay'][0] == 0 and \
#    df_test['Combined_Arr_Delay'][0] == -24 and \
#    df_test['Can_Status'][0] == 0 and \
#    df_test['Scheduled_Elapsed_Time'][0] == 120 and \
#    df_test['Actual_Elapsed_Time'][0] == 97    