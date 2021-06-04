# Project Overview
This project contains code for the atnresilience project. The project uses data-driven metrics and graph theory to analyze the US Air Transportation Network (ATN), focusing on analyzing disruptions (e.g., extreme weather events) and their impact on the resilience of the overal US ATN and individual airline networks.

# Installing
- Requires Python 3.
- Requires [Basemap](https://matplotlib.org/basemap/users/installing.html).
- See `environment.yml` file for other required packages.
- [DB Browser for SQLite](http://sqlitebrowser.org) is recommended to view SQLite database with a GUI.

# Testing
- The `/tests` directory has 4 test scripts which tests functions which have returns (not graphs).
- The first test script will create a test sqlite database from the test data in `/tests/raw` directory.
- The fourth test script deletes the test database to prevent conflicts if pytest is run multiple times.
- Running `pytest` in the root atnresilience directory may cause errors. Use `python -m pytest tests/` in the root directory to resolve this issue. 
- The tests will return warnings due to the square root of negative numbers. This is expected and is due to the small size of the test data set.

# Contributing
- See the contribution guide on [myproject](https://github.com/Tran-Research-Group/myproject.git)

# Running Scripts
- Before running the scripts, create the data directory and its subdirectories. 
	- `/data/` under the root directory.
	- `/data/raw/` for raw data. Raw data must follow the format 'yyyymm'.
	- `/data/processed/` where the database and processed data will be saved.
- All functions can be run from three jupyter notebooks in the following order:
	1. `/notebooks/atn_db.ipynb` creates the sqlite database used for calculations from raw data in the `/data/raw` directory.
	2. `/notebooks/atn_analysis.ipynb` creates csv files of z-score and Mahalanobis distance to the `/data/processed` directory.
	3. `/notebooks/atn_visualization.ipynb` contains several functions used for the visualization of the data.

# License
University of Illinois <br />
Open Source License <br />

Copyright © 2019, Intelligent Systems and Analytics Laboratory, University of Illinois. All rights reserved. <br />

Developed by: <br />
Tran Research Group <br />
Department of Aerospance Engineering <br />
University of Illinois at Urbana-Champaign <br />
https://tran.aerospace.illinois.edu <br />

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: <br />
Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers.
Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution. <br />
Neither the names of <Name of Development Group, Name of Institution>, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission. <br />
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
