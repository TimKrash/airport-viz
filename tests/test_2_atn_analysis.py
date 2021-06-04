from atnresilience import atn_analysis
import sqlite3
import pandas as pd

raw_path = 'tests/raw/'
processed_direc = 'tests/processed/'
db_path = 'tests/processed/test_db.sqlite'

def test_mahalanobis_distance():
    atn_analysis.mahalanobis_distance(db_path,2015,'WN','ADM',processed_direc)
    
    #Expected values to be compared against
    md_expected =[0.72331, 1.47892, 2.55176, 5.07912, 0.40969, 0.90910, 0.26665, 0.32672, 0.91671, 0.18855]
    
    #Get the calculated values to test against. The first 10 values will be tested 
    md_data_path = '%s2015_WN_MDdata_ADM.csv' %(processed_direc)
    md_df = pd.read_csv(md_data_path)    
    md_test = (md_df.head(10))['Arrival_Delay'].tolist()
    
    #Test that the calculated values match the expected.
    assert [round(n,5) for n in md_test] == md_expected
    
def test_z_score():
    atn_analysis.z_score(db_path,2015,'WN','ADM',processed_direc)
    
    #Expected values to be compared against
    atl_expected = [0.28202, 0.96852, 0.99921, 0.99999, 0.62891, 0.86051, 0.39747, 0.28138, 0.54704, 0.41336]
    lax_expected = [0.55227, 0.57571, 0.84268, 0.71301, 0.46171, 0.66430, 0.47509, 0.15734, 0.13177, 0.48330]
    
    #Get the calculated values to test against. The first 10 values for each of the 2 airports will be tested 
    z_data_path = '%s2015_WN_Zdata_ADM.csv' %(processed_direc)
    z_df = pd.read_csv(z_data_path)
    atl_test = (z_df.head(10))['ATL'].tolist()
    lax_test = (z_df.head(10))['LAX'].tolist()
        
    #Test that the calculated values match the expected for both airports in the data set.
    assert [round(n,5) for n in atl_test] == atl_expected and\
    [round(n,5) for n in lax_test] == lax_expected
    
    
