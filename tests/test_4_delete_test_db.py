import os

db_path = 'tests/processed/test_db.sqlite'

def test_delete_test_db():
    #This should always be the last script run. This will delete the test db created to avoid appending onto it on future tests
    os.remove(db_path)    