from atnresilience import atn_tools as tools

db_path = 'tests/processed/test_db.sqlite'
processed_direc = 'tests/processed/'

#import os
#dir_path = os.path.dirname(os.getcwd())
#script_dir = dir_path + '/atnresilience'
#os.chdir(script_dir)
#
#import atn_tools as tools
#
#processed_direc = 'tests/processed/'
#db_path = dir_path + '/tests/processed/test_db.sqlite'

def test_weighted_edge():
    
    origin_top_expected = ['ABQ', 'ATL', 'AUS', 'AUS', 'BDL', 'BNA', 'BOS', 'BWI', 'BWI', 'CAK',]
    weight_top_expected = [2, 2, 2, 2, 2, 2, 2, 3, 2, 2]
    
    origin_bottom_expected = ['SJU', 'SLC', 'SMF', 'STL', 'STL', 'TPA', 'TUS',]
    weight_bottom_expected = [1, 2, 3, 2, 2, 3, 2]
    
    weight_df = tools.weighted_edge(db_path, 2015, 'WN')
    
    origin_top = (weight_df.head(10))['Origin'].tolist()
    weight_top = (weight_df.head(10))['Weight'].tolist()
    origin_bottom = (weight_df.tail(7))['Origin'].tolist()
    weight_bottom = (weight_df.tail(7))['Weight'].tolist()
    
#    return(weight_df.tail(7))
    assert origin_top == origin_top_expected and\
    weight_top == weight_top_expected and\
    origin_bottom == origin_bottom_expected and\
    weight_bottom == weight_bottom_expected
    
def remove_frequency():
    remove_dict = remove_frequency(db_path, 2015, 'WN', 'ADM', 0.1, 0.95, processed_direc)
    
    assert remove_dict['BOX'] == 0 and\
    remove_dict['CLT'] == 31 and\
    remove_dict['DEN'] == 189 and\
    remove_dict['DFW'] == 132 and\
    remove_dict['ELP'] == 28 and\
    remove_dict['JFK'] == 0 and\
    remove_dict['LAX'] == 33 and\
    remove_dict['MSP'] == 31 and\
    remove_dict['OFF'] == 29 and\
    remove_dict['OMA'] == 21 and\
    remove_dict['PSP'] == 12