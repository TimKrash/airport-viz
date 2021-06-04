import os
import math
import copy
import random
import calendar
import csv
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import sqlite3
import seaborn as sns

#from atnresilience import atn_analysis as atn
import atn_analysis
import db_tools

# Set global styles for plots
plt.rcParams["font.family"] = "Times New Roman"
sns.set_palette("colorblind")
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8)
line_type = {1:'-',2:'--',3:':',4:'-.'} 

def remove_frequency(db_path, file, airline, include_data, can_limit, zs_limit, processed_direc):
    """
    Creates a dictionary of airports and their removal frequency for a given airline

    Parameters
    ----------
    file: int
        Year of selected data
    airline: string
        Airline to get data from
    include_data: string
        Type of airline data to query from csv
    can_limit: int
        Cancellation limit 
    zs_limit: int
        The z-score limit
    
    Returns
    -------
    Returns a dictionary containing airport removal frequency values  
    
    Notes
    -----
    
    """

    df_net_tuple = pd.DataFrame() 
    
    df_net = atn_analysis.raw_query(db_path, file, airline)

    df_net_tuple["Origin"] = df_net.Origin_Airport_Code
    df_net_tuple["Destination"] = df_net.Destination_Airport_Code

    graph = [tuple(x) for x in df_net_tuple.to_records(index=False)]
    G = nx.Graph()

    G.add_edges_from(graph)

    tempG = G.copy()

    Airport_Dict = {}
    for i in G.nodes():
        Airport_Dict[i] = 0

    Total_List = get_remove_list(db_path, file,include_data, airline, can_limit, zs_limit, processed_direc)

    
    if int(file)%4 == 0:
        total_day = 366
    else:
        total_day = 365
        
    for j in range(total_day):
        airport_list = Total_List[j]
        for l in airport_list:
            tempG.remove_node(l)

            Airport_Dict[l] = Airport_Dict[l] + 1

        tempG = G.copy()

    return(Airport_Dict)

def weighted_edge(db_path, file, airline):
    """
    Creates a data frame of origin airports, destination airports and weights for each route

    Parameters
    ----------
    file: int
        Year of selected data
    airline: string
        Airline to get data from
    include_data: string
        Type of airline data to query from csv
    can_limit: int
        Cancellation limit 
    zs_limit: int
        The z-score limit
    
    Returns
    -------
    Returns a data frame containing each respective weighted route from an origin airport to a destination
    
    Notes
    -----
    
    """
    df = atn_analysis.raw_query(db_path, file, airline)
    by_origin = df.groupby([df.Origin_Airport_Code]).Can_Status.count()
    airport_list = by_origin.index.tolist()
    df = df[df['Destination_Airport_Code'].isin(airport_list)]
    
    df_tuple = pd.DataFrame()
    df_weighted = df.groupby([df.Origin_Airport_Code, df.Destination_Airport_Code]).Can_Status.count().reset_index()
    df_tuple["Origin"] = df_weighted.Origin_Airport_Code
    df_tuple["Destination"] = df_weighted.Destination_Airport_Code
    file_str = int(str(file)[:4])
    
    if calendar.isleap(file_str) == 1:
        days = 366
    else:
        days = 365
    
    df_tuple["Weight"] = df_weighted.Can_Status
    
    weight_values = [math.log(y, 10) for y in df_tuple.Weight.values]
    for i in range(0, len(weight_values)):
        df_tuple.Weight.values[i] = weight_values[i]
    
    return(df_tuple) 


def get_remove_list(db_path, file, include_data, airline, can_limit, zs_limit, processed_direc):
    """
    Return a remove_list in a year (airline specific, include_data specific) based on cancelation limit and z_score limit.  
    
    Parameters
    ----------
    file: int
        Year of selected data
    include_data: string
        Specify what kind of data to include in processed flight data. See drop_flights in M-D File. Possible parameters are:
            CC: Cancellations only
            ADD: Arrival delays including diversions
            ADM: Purely arrival delays excluding cancellations or diversions
            DCC: Combined delay. If arrival delay is greater than a set threshold, the flight is considered cancelled
            DD: Departure delays. Does not include cancelled or diverted flights.
    airline: string
        Airline to get data from. This is the 2 letter airline code (ex: AA, UA, DL, WN)
    can_limit: float
        Cancellation Limit. Between 0 and 1
    zs_limit: float
        z-score limit. Between 0 and 1
    
    Returns
    -------
    Pandas df 
        
    Notes
    -----
    
    """

    z_score_path = '%s%s_%s_Zdata_%s.csv'%(processed_direc, file,airline,include_data)  
    #df_score = pd.read_csv(raw_file_drop, index_col="Date")
    df_score = pd.read_csv(z_score_path, index_col = "Day_of_Year")
    df_score.index = pd.to_datetime(df_score.index)
    airport_list = df_score.columns.tolist()
    
    df = atn_analysis.raw_query(db_path,file,airline)

    df = df[df['Origin_Airport_Code'].isin(airport_list)]  # Filtering to make sure airports are equal in both directions
    df = df[df['Destination_Airport_Code'].isin(airport_list)]
    by_origin_count = df.groupby(['Flight_Date', 'Origin_Airport_Code'], as_index=False)[['Can_Status']].count()
    by_origin = df.groupby(['Flight_Date', 'Origin_Airport_Code'], as_index=False)[['Can_Status']].sum()
    by_origin.Can_Status = by_origin.Can_Status / by_origin_count.Can_Status
    #print(by_origin)
    df_score["idx"] = df_score.index
    df_score = pd.melt(df_score, id_vars='idx', value_vars=airport_list)
    df_score = df_score.sort_values(['idx', 'variable'], ascending=[True, True])
    df_score.columns = ["Date", "Airports", "Z_Score"]
    df_score.set_index('Date')
    df_score["Cancellations"] = by_origin.Can_Status

    ### Creating the or conditions. First is the percentage of delayed flights and the second is the z-score
    df_score["Z_score_9901"] = np.where((df_score['Cancellations'] > can_limit) | (df_score['Z_Score'] > zs_limit), 1, 0)
    #print(df_score)

    ### Creating pivot table for easy manipulation. This creates the date as the index with the properties corresponding to
    ### it and finally repeats this trend for all airports being considered.
    df_pivot = df_score.pivot_table('Z_score_9901', ['Date'], 'Airports')
    #print(df_pivot)

    s = np.asarray(np.where(df_pivot == 1, ['{}'.format(x) for x in df_pivot.columns], '')).tolist()


    s_nested = []
    for k in s:
        p = list(filter(None,k))
        
        #p = filter(None,k)
        s_nested.append(p)
        #s_nested.extend(p)


    return s_nested


def inv_average_shortest_path_length(graph, weight=None):
    """
    Creates an unweight inverse average path length graph

    Parameters
    ----------
    graph: python graph object
    weight: default

    Returns
    -------
    Returns the IAPL unweighted graph
        
    Notes
    -----

    """
    avg = 0.0
    if weight is None:
        for node in graph:
            avg_path_length = nx.single_source_shortest_path_length(graph, node)    # get the shortest path lengths from source to all reachable nodes (unweighted)
            del avg_path_length[node]  # Deletes source node from the list to avoid division by 0
            inv_avg_path_length = copy.deepcopy(avg_path_length)
            inv_avg_path_length.update((x, 1/y) for x, y in avg_path_length.items())
            avg += sum(inv_avg_path_length.values())
            
    
    n = len(graph)

    if n == 1 or n == 0:
        return 0
    else:
        return avg/(n*(n-1))


def inv_average_shortest_path_length_W(graph, weight=None):
    """
    Creates the table atn_performance in the database at the specified input location if one does not exist.

    Parameters
    ----------
    graph: python graph object
    weight: default

    Returns
    -------
    Returns the inverse average path length weighted graph
        
    Notes
    -----

    """

    avg = 0.0
    if weight is None:
        for node in graph:
            avg_path_length = nx.single_source_dijkstra_path_length(graph, node)  # get the shortest path lengths from source to all reachable nodes (weighted)
            del avg_path_length[node]  # Deletes source node from the list to avoid division by 0
            inv_avg_path_length = copy.deepcopy(avg_path_length)
            inv_avg_path_length.update((x, 1/y) for x, y in avg_path_length.items())
            avg += sum(inv_avg_path_length.values())
    
    n = len(graph)

    if n == 1 or n == 0:
        return 0
    else:
        return avg/(n*(n-1))    
        

def Data_Driven_W(file_list, airline_list, include_data, can_limit, zs_limit, processed_direc, graph_direc):
    """
    Calculate the cluster size and IAPL for each day in a year after removal based on data-driven method. 

    Parameters
    ----------
    file_list: list
        List contaning years to process
    airline_list: list
        List contaning airlines to process
    include_data: string
        Specify what kind of data to include in processed flight data. See drop_flights in M-D File. Possible parameters are:
            CC: Cancellations only
            ADD: Arrival delays including diversions
            ADM: Purely arrival delays excluding cancellations or diversions
            DCC: Combined delay. If arrival delay is greater than a set threshold, the flight is considered cancelled
            DD: Departure delays. Does not include cancelled or diverted flights.
    can_limit: float
        Cancellation threshold
    zs_limit: float
        z-score threshold

    Returns
    -------
    The cluster size and IAPL for each day of the year after removal based on data-driven method.
        
    Notes
    -----

    """
    for file in file_list:
##    iteration of years first
        figure_num = 1
        CSV_df = pd.DataFrame(columns = airline_list)
        for airline in airline_list:
            # CSV_df[airline] = [1,2,3,4]
            # CSV_file = "%s_DD_IAPL.csv" %(file)
            # CSV_df.to_csv(CSV_file, index=False)



##    Get the directory path
            script_dir = os.path.dirname(os.getcwd())
            db_local_path = "data/processed/atn_db.sqlite"

##    df set up from Keshav (NO CHANGE) (Weighted Graph)
            df = pd.DataFrame()
            db_path = os.path.join(script_dir, db_local_path)
            fields = ["Origin_Airport_Code", "Destination_Airport_Code", "Can_Status"]
            df_net = atn_analysis.raw_query(db_path,file,airline)

            df["Origin_Airport_Code"] = df_net.Origin_Airport_Code
            df["Destination_Airport_Code"] = df_net.Destination_Airport_Code
            df["Can_Status"] = df_net.Can_Status

            by_origin = df.groupby([df.Origin_Airport_Code]).Can_Status.count()
            airport_list = by_origin.index.tolist()
            df = df[df['Destination_Airport_Code'].isin(airport_list)]
            #print(df)
            df_tuple = pd.DataFrame()
            df_weighted = df.groupby([df.Origin_Airport_Code, df.Destination_Airport_Code]).Can_Status.count().reset_index()
            df_tuple["Origin"] = df_weighted.Origin_Airport_Code
            df_tuple["Destination"] = df_weighted.Destination_Airport_Code

            if int(file)%4 == 0:
                days = 366
            else:
                days = 365

            df_tuple["Weight"] = df_weighted.Can_Status/days
            df_tuple.Weight = 1/df_tuple.Weight

##    Output lists initialization:
            #day_IAPL = 0
            day_CS = 0
            #output_IAPL = []
            output_CS = []
            NoD = []

##    Graph object initialization
            graph = [tuple(x) for x in df_tuple.to_records(index=False)]
            G = nx.Graph()

##    Set up the weighted graph
            G.add_weighted_edges_from(graph)
            #print(G.nodes())
            
            tempG = G.copy()   #use temporary graph for the loop

##    Remove list for the whole year
            Total_Remove_List = get_remove_list(db_path,file,include_data, airline, can_limit, zs_limit,processed_direc) 
            
            if int(file)%4 == 0:
                total_day = 366
            else:
                total_day = 365
                
            for j in range(total_day):
##    Remove the nodes in each day and get the CS and IAPL data

                #day_IAPL = 0
                Day_Remove_List = Total_Remove_List[j]
                
                NoD.append(j)
                
                for l in Day_Remove_List:
                    tempG.remove_node(l)
                    #largest_component_b = max(nx.connected_components(tempG), key=len)
                    
                #day_IAPL =(inv_average_shortest_path_length_W(tempG))
                largest_component_b = max(nx.connected_components(tempG), key=len)
                day_CS = len(largest_component_b)    
                #len(largest_component_b) = cluster size
                #cluster fraction = cluster size/number of nodes    

                #output_IAPL.append(day_IAPL)
                output_CS.append(day_CS)
                #sum_IAPL = sum_IAPL + (inv_average_shortest_path_length(tempG))
                tempG = G.copy()

##    plotting command
            plt.figure(figure_num)
            #line = plt.plot(NoD,output_IAPL, label="{}".format(airline))
            line = plt.plot(NoD,output_CS, label="{}".format(airline))
            plt.legend()

            #CSV_df[airline] = output_IAPL
            CSV_df[airline] = output_CS
            
        #CSV_file = "%s_DD_IAPL.csv" %(file)
        CSV_file = "%s%s_DD_CS.csv" %(graph_direc,file)
        CSV_df.to_csv(CSV_file, index=False)


        #plt.title("{} Data Driven IAPL".format(str(file)))
        plt.xlabel("Day")
        #plt.ylabel("IAPL")
        plt.ylabel("Cluster Size")    
        #plt.savefig("{}_Data_Driven_IAPL.png".format(str(file)))
        plt.savefig("%s%s_Data_Driven_CS.png"%(graph_direc,file))
        plt.show()
        figure_num = figure_num + 1

def Pure_Graph_W_Shu(file_list, airline_list, include_data, processed_direc, rep_num):
    """
    Calculate the linear algebraic connectivity, cluster size and IAPL for each day in a year after random removal based on Pure Graph method. 
    Random Removal set up by shuffle function

    Parameters
    ----------
    file_list: list
        List contaning years to process
    airline_list: list
        List contaning airlines to process
    include_data: string
        Specify what kind of data to include in processed flight data. See drop_flights in M-D File. Possible parameters are:
            CC: Cancellations only
            ADD: Arrival delays including diversions
            ADM: Purely arrival delays excluding cancellations or diversions
            DCC: Combined delay. If arrival delay is greater than a set threshold, the flight is considered cancelled
            DD: Departure delays. Does not include cancelled or diverted flights.
    rep_num: int
        Number of repititions


    Returns
    -------
    csv with the cluster size and IAPL for each day of the year after removal based on data-driven method.
        
    Notes
    -----

    """

    
    for airline in airline_list:
        rep_ite = 1
        Total_AC = []
        Total_Cluster_Size = []
        Total_IAPL = []
        for i in range(len(file_list)):
##    initialize the output lists
            Total_AC.append(0)
            Total_Cluster_Size.append(0)
            Total_IAPL.append(0)

##    Save the data in csv
        filename1 = "%s%s_ACR.csv" %(processed_direc,airline)
        with open(filename1, 'w') as myfile1:
            wr1 = csv.writer(myfile1, quoting=csv.QUOTE_ALL)
            wr1.writerow(file_list)

        filename2 = "%s%s_IAPLR.csv" %(processed_direc,airline)
        with open(filename2, 'w') as myfile2:
            wr2 = csv.writer(myfile2, quoting=csv.QUOTE_ALL)
            wr2.writerow(file_list)

        filename3 = "%s%s_CSR.csv" %(processed_direc,airline)
        with open(filename3, 'w') as myfile3:
            wr3 = csv.writer(myfile3, quoting=csv.QUOTE_ALL)
            wr3.writerow(file_list)

        while rep_ite < rep_num+1:
##    start the reptition

            year_IAPL = []
            year_Cluster_Size = []
            year_AC = []
            for file in file_list:
##    Get the directory path
                script_dir = os.path.dirname(os.getcwd())
                db_local_path = "data/processed/atn_db.sqlite"

##    df set up from Keshav (NO CHANGE)
                df = pd.DataFrame()
                db_path = os.path.join(script_dir, db_local_path)
                fields = ["Origin_Airport_Code", "Destination_Airport_Code", "Can_Status"]
                #df_net = pd.read_csv(comb_file, usecols=fields)
                df_net = atn_analysis.raw_query(db_path,file,airline)

                df["Origin_Airport_Code"] = df_net.Origin_Airport_Code
                df["Destination_Airport_Code"] = df_net.Destination_Airport_Code
                df["Can_Status"] = df_net.Can_Status

                by_origin = df.groupby([df.Origin_Airport_Code]).Can_Status.count()
                airport_list = by_origin.index.tolist()
                df = df[df['Destination_Airport_Code'].isin(airport_list)]
                #print(df)
                df_tuple = pd.DataFrame()
                df_weighted = df.groupby([df.Origin_Airport_Code, df.Destination_Airport_Code]).Can_Status.count().reset_index()
                df_tuple["Origin"] = df_weighted.Origin_Airport_Code
                df_tuple["Destination"] = df_weighted.Destination_Airport_Code

                if int(file)%4 == 0:
                    days = 366
                else:
                    days = 365

                df_tuple["Weight"] = df_weighted.Can_Status/days
                df_tuple.Weight = 1/df_tuple.Weight

##    Output lists initialization:


##    Graph object initialization
                graph = [tuple(x) for x in df_tuple.to_records(index=False)]
                G = nx.Graph()

                G.add_weighted_edges_from(graph)
                NodeNum = G.number_of_nodes()
                #print('Weighted Alebraic Connectivity: ', nx.algebraic_connectivity(G))
                year_AC.append(nx.algebraic_connectivity(G))

                sum_IAPL = 0
                sum_Cluster_Size = 0
                IAPL_list = []
                Cluster_Size_list = []
                Remove_List = []

                for node in G.nodes():
##    Get the list of the airports
                    Remove_List.append(node)
                
##    Shuffle the lists
                random.shuffle(Remove_List)
                for l in Remove_List:
                    G.remove_node(l)
                    if len(G.nodes()) != 0:

##    Add up the data after removing each node
                        largest_component_b = max(nx.connected_components(G), key=len)
                        IAPL_list.append(inv_average_shortest_path_length_W(G))
                        Cluster_Size_list.append(len(largest_component_b)/NodeNum)            
                        sum_IAPL = sum_IAPL + (inv_average_shortest_path_length_W(G))
                        sum_Cluster_Size = sum_Cluster_Size + len(largest_component_b)/NodeNum
                
##    Save the data of the year
                year_IAPL.append(sum_IAPL)
                year_Cluster_Size.append(sum_Cluster_Size)

            with open(filename1, 'a') as myfile1:
                wr1 = csv.writer(myfile1, quoting=csv.QUOTE_ALL)
                wr1.writerow(year_AC)

            with open(filename2, 'a') as myfile2:
                wr2 = csv.writer(myfile2, quoting=csv.QUOTE_ALL)
                wr2.writerow(year_IAPL)

            with open(filename3, 'a') as myfile3:
                wr3 = csv.writer(myfile3, quoting=csv.QUOTE_ALL)
                wr3.writerow(year_Cluster_Size)


            # print('Unweighted Summation of IAPL: ', sum_IAPL)
            # print('Unweighted Summation of Cluster Size: ', sum_Cluster_Size)
            # print('Unweighted IAPL list', IAPL_list)
            for i in range(len(file_list)):
##    Get the sum for the average
                Total_AC[i] = Total_AC[i] + year_AC[i]
                Total_IAPL[i] = Total_AC[i] + year_IAPL[i]
                Total_Cluster_Size[i] = Total_Cluster_Size[i] + year_Cluster_Size[i]

            rep_ite = rep_ite + 1


        for i in range(len(file_list)):
##    Get the average
            Total_AC[i] = Total_AC[i]/rep_num
            Total_IAPL[i] = Total_IAPL[i]/rep_num
            Total_Cluster_Size[i] = Total_Cluster_Size[i]/rep_num


##    Plotting Command:
        plt.figure(num=1,figsize=(2.8,2.0),dpi=300)
#        line1 = plt.plot(file_list,Total_IAPL, label="{}".format(airline))
        plt.plot(file_list,Total_IAPL, label="{}".format(airline))
        plt.legend()
        
        plt.figure(num=2,figsize=(2.8,2.0),dpi=300)
#        line2 = plt.plot(file_list,Total_Cluster_Size, label="{}".format(airline))
        plt.plot(file_list,Total_Cluster_Size, label="{}".format(airline))
        plt.legend()

        plt.figure(num=3,figsize=(2.8,2.0),dpi=300)
#        line3 = plt.plot(file_list,Total_AC, label="{}".format(airline))
        plt.plot(file_list,Total_AC, label="{}".format(airline)) 
        plt.legend()
        
    plt.figure(1)
    plt.title("IAPL (Random)")
    plt.xlabel("Year")
    plt.ylabel("IAPL")
    plt.savefig("Pure_Graph_IAPLR.png")

    plt.figure(2)
    plt.title("Cluster Size (Random)")
    plt.xlabel("Year")
    plt.ylabel("Cluster Size")
    plt.savefig("Pure_Graph_CSR.png")

    plt.figure(3)
    plt.title("Algebraic Connectivity (Random)")
    plt.xlabel("Year")
    plt.ylabel("Algebraic Connectivity")
    plt.savefig("Pure_Graph_ACR.png")

    plt.show()


def Pure_Graph_W_Tar(file_list,airline_list,processed_direc,graph_direc):
    """
    Calculate the linear algebraic connectivity, cluster size and IAPL for each day in a year after targeted removal based on Pure Graph method.
    Targeted removal set up by the degree of the nodes. (Remove the node with higher node first, degree calculated when the weight is set as flight frequency)
    
    Parameters
    ----------
    file_list: list
        List contaning years to process
    airline_list: list
        List contaning airlines to process

    Returns
    -------
    Graph with the removels.
        
    Notes
    -----

    """

    line_type_iter = 0
    for airline in airline_list:
        line_type_iter += 1
        year_IAPL = []
        year_Cluster_Size = []
        year_AC = []

        for file in file_list:
##    Get the directory path
            script_dir = os.path.dirname(os.getcwd())
            #comb_path = "data/processed/%s_%s_combined.csv" % (file,airline)
            db_local_path = "data/processed/atn_db.sqlite"

##    df set up from Keshav (NO CHANGE)
            df = pd.DataFrame()
            db_path = os.path.join(script_dir, db_local_path)
            fields = ["Origin_Airport_Code", "Destination_Airport_Code", "Can_Status"]
            #df_net = pd.read_csv(comb_file, usecols=fields)
            df_net = atn_analysis.raw_query(db_path,file,airline)

            df["Origin_Airport_Code"] = df_net.Origin_Airport_Code
            df["Destination_Airport_Code"] = df_net.Destination_Airport_Code
            df["Can_Status"] = df_net.Can_Status

            by_origin = df.groupby([df.Origin_Airport_Code]).Can_Status.count()
            airport_list = by_origin.index.tolist()
            df = df[df['Destination_Airport_Code'].isin(airport_list)]
            #print(df)
            df_tuple = pd.DataFrame()
            df_weighted = df.groupby([df.Origin_Airport_Code, df.Destination_Airport_Code]).Can_Status.count().reset_index()
            df_tuple["Origin"] = df_weighted.Origin_Airport_Code
            df_tuple["Destination"] = df_weighted.Destination_Airport_Code

            if int(file)%4 == 0:
                days = 366
            else:
                days = 365

            df_tuple["Weight"] = df_weighted.Can_Status/days
            
##    Graph object initialization
            graph = [tuple(x) for x in df_tuple.to_records(index=False)]
            G1 = nx.Graph()

            G1.add_weighted_edges_from(graph)
            #print('Weighted Alebraic Connectivity: ', nx.algebraic_connectivity(G))

            sum_IAPL = 0
            sum_Cluster_Size = 0
            IAPL_list = []
            Cluster_Size_list = []
            Remove_List = []


            tempG = G1.copy()
#            return(tempG.nodes())
##    Get the remove list based on the node degree
            while list(tempG.nodes()) != []:
#                print('run')
                MaxNode = list(tempG.nodes())[0]
                MaxDegree = tempG.degree(MaxNode)
                for node in tempG.nodes():
                    if tempG.degree(node) >= MaxDegree:
                        MaxNode = node
                        MaxDegree = tempG.degree(node)
                    
                tempG.remove_node(MaxNode)
                Remove_List.append(MaxNode)

            #print('Ordered List: ', Remove_List)


            df_tuple.Weight = 1/df_tuple.Weight

            graph = [tuple(x) for x in df_tuple.to_records(index=False)]
            G = nx.Graph()

            G.add_weighted_edges_from(graph)
            year_AC.append(nx.algebraic_connectivity(G))
            #print('Weighted Alebraic Connectivity: ', nx.algebraic_connectivity(G))
            NodeNum = G.number_of_nodes()

## add on the data after every removal
            for l in Remove_List:
                G.remove_node(l)
                if list(G.nodes()) != []:
                    largest_component_b = max(nx.connected_components(G), key=len)
                    IAPL_list.append(inv_average_shortest_path_length_W(G))
                    Cluster_Size_list.append((len(largest_component_b))/NodeNum)            
                    sum_IAPL = sum_IAPL + (inv_average_shortest_path_length_W(G))
                    sum_Cluster_Size = sum_Cluster_Size + (len(largest_component_b))/NodeNum

            year_IAPL.append(sum_IAPL)
            year_Cluster_Size.append(sum_Cluster_Size)

        plt.figure(1,figsize=(2.8,2.0),dpi=300)
        line1 = plt.plot(file_list,year_IAPL, label="{}".format(airline),linestyle=line_type[line_type_iter], marker = 'o')
        plt.xticks(file_list)
        plt.xlabel('Year',fontsize=10)
        plt.ylabel('IAPL',fontsize=10)
        plt.legend()
#        fig1, ax1 = plt.subplots(figsize=(2.8,1.8))
#        sns.lineplot(ax=ax1,x=file_list,y=year_IAPL,label="{}".format(airline))
#        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.figure(2,figsize=(2.8,2.0),dpi=300)
        line2 = plt.plot(file_list,year_Cluster_Size, label="{}".format(airline),linestyle=line_type[line_type_iter], marker = 'o')
        plt.xticks(file_list)
        plt.xlabel('Year',fontsize=10)
        plt.ylabel('Cluster Size',fontsize=10)
        plt.legend()
#        fig2, ax2 = plt.subplots(figsize=(2.8,1.8))
#        sns.lineplot(ax=ax2,x=file_list,y=year_Cluster_Size,label="{}".format(airline))
#        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))



        plt.figure(3,figsize=(2.8,2.0),dpi=300)
        line3 = plt.plot(file_list,year_AC, label="{}".format(airline),linestyle=line_type[line_type_iter], marker = 'o')
        plt.xticks(file_list)
        plt.xlabel('Year',fontsize=10)
        plt.ylabel('Algebraic Connectivity',fontsize=10)
        plt.legend()

##    Save the data    
        filename1 = "%s%s_ACT.csv" %(processed_direc,airline)
        with open(filename1, 'w') as myfile1:
            wr1 = csv.writer(myfile1, quoting=csv.QUOTE_ALL)
            wr1.writerow(file_list)
            wr1.writerow(year_AC)

        filename2 = "%s%s_IAPLT.csv" %(processed_direc,airline)
        with open(filename2, 'w') as myfile2:
            wr2 = csv.writer(myfile2, quoting=csv.QUOTE_ALL)
            wr2.writerow(file_list)
            wr2.writerow(year_IAPL)

        filename3 = "%s%s_CST.csv" %(processed_direc,airline)
        with open(filename3, 'w') as myfile3:
            wr3 = csv.writer(myfile3, quoting=csv.QUOTE_ALL)
            wr3.writerow(file_list)
            wr3.writerow(year_Cluster_Size)

    plt.figure(1)
    #plt.title("IAPL (Target)")
#    plt.xlabel("Year")
#    plt.ylabel("IAPL")
    IAPLT_path = "%starget_IAPL.pdf"%(graph_direc,)
    plt.tight_layout()
    plt.legend(labelspacing=0.15,fontsize=10)
    plt.savefig(IAPLT_path)
    print('Targeted IAPL graph saved to %s'%(IAPLT_path,))


    plt.figure(2)
    #plt.title("Cluster Size (Target)")
#    plt.xlabel("Year")
#    plt.ylabel("Cluster Size")
    CST_path = "%starget_CST.pdf"%(graph_direc,)
    plt.tight_layout()
    plt.legend(labelspacing=0.15,fontsize=10)
    plt.savefig(CST_path)
    print('Targeted CS graph saved to %s'%(CST_path,))


    plt.figure(3)
    #plt.title("Algebraic Connectivity (Target)")
#    plt.xlabel("Year")
#    plt.ylabel("Algebraic Connectivity")
    ACT_path = "%starget_ACT.pdf"%(graph_direc,)
    plt.tight_layout()
    plt.legend(labelspacing=0.15,fontsize=10)
    plt.savefig(ACT_path)
    print('Targeted AC graph saved to %s'%(ACT_path,))

#    plt.show()       
       
def ran_remove_shaded(year_list,airline_list,processed_direc,graph_direc):
    """
    Creates a the shaded lineplot for the random removals - cluster size, algebraic connecticity, IAPL.

    Parameters
    ----------
    year_list: list
        List of tears to plot
    airline_list: list
        List of airlines to plot
    
    Returns
    -------
    Saves the plots to the /data/graphs folder in pdf format.
    
    Notes
    -----
    
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


    plt.figure(1,figsize=(2.8,2.0),dpi=300)
    ax1 = sns.lineplot(data=CS_df_all, x = 'Year', y = 'Cluster_Size', hue='Airline', style='Airline', marker = 'o')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel('Year')
    plt.ylabel('Cluster Size')
    plt.legend(airline_list,fontsize=10,labelspacing=0.15)
    plt.tight_layout()
    plt.savefig('%sShaded_CS.pdf'%(graph_direc,))

    plt.figure(2,figsize=(2.8,2.0),dpi=300)
    ax2 = sns.lineplot(data=IAPL_df_all, x = 'Year', y = 'IAPL', hue='Airline', style='Airline', marker = 'o')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel('Year')
    plt.ylabel('IAPL')
    plt.legend(airline_list,fontsize=10,labelspacing=0.15)
    plt.tight_layout()
    plt.savefig('%sShaded_IAPL.pdf'%(graph_direc,))

    plt.figure(3,figsize=(2.8,2.0),dpi=300)
    ax3 = sns.lineplot(data=AC_df_all, x = 'Year', y = 'AC', hue='Airline', style='Airline', marker = 'o')
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.xlabel('Year')
    plt.ylabel('Algebraic Connectivity')
    plt.legend(airline_list,fontsize=10,labelspacing=0.15)
    plt.tight_layout()
    plt.savefig('%sShaded_AC.pdf'%(graph_direc,))

    plt.show()  


def dropped_metrics(db_path,year,airline,include_data):
    """
    Creates a csv with delay metrics including the number and percentage of dropped airports 
    and flights and corresponding columns of data based on the include_data parameter 
    
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
    
    Returns
    -------
    df : dataframe
        Pandas dataframe of the metrics

    
    Notes
    -----
    Creates a csv of airports and relevant data used for M-D and Z-score for each day depending on the data sorting you specified. 
    The atn db must be created first. Please see create_atn_db.
    
    """
    
    #include_dict is a dictionary which maps the given include_data to the SQL statement that that parameter corresponds to. Since DCC has an extra condition, it is seperate. 
    include_dict = {
        "ADD" : "Can_Status != 1 AND Combined_Arr_Delay IS NOT NULL AND",
        "ADM" : "Can_Status != 1 AND Div_Status != 1 AND",
        "CC" : "",
        "DD" : "Can_Status != 1 AND Dep_Delay IS NOT NULL AND",
        "DCC" : "Combined_Arr_Delay IS NOT NULL AND"
        } 
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
    #Total rows = total number of flights
    total_flights = len(df_int)
    for day in unique_days:
        df_day = df_int[df_int.Day_Of_Year == day]
        #day_sets is a list of sets. Each sets includes all the airports for that day
        if include_data in ["CC","DD",]:
            day_sets += [set(df_day.Origin_Airport_Code)] 
        else:
            day_sets += [set(df_day.Destination_Airport_Code)] 

    relevant_airports = set.intersection(*day_sets)
    relevant_airports = sorted(relevant_airports)
    
    # Run the relevant dropped metric calculations    
    all_airports = df_int['Destination_Airport_Code'].unique()
    dropped_percent = 1- (len(relevant_airports)/len(all_airports))
    
    if include_data in ["CC","DD",]:
        df_dropped = df_int.reset_index().set_index(['Origin_Airport_Code']).sort_index()
    else:
        df_dropped = df_int.reset_index().set_index(['Destination_Airport_Code']).sort_index()
    
    df_dropped = df_dropped.loc[relevant_airports]
    df_dropped = df_dropped.reset_index().set_index(['index']).sort_index()
    
    num_dropped = total_flights - len(df_dropped)
    flight_drop_percent = num_dropped/total_flights
    
    dropped_metrics = [year, airline, dropped_percent, len(all_airports), len(relevant_airports),total_flights,num_dropped,flight_drop_percent]
    return(dropped_metrics)    

def calculate_betweenness_centrality(year,airline):
    '''
    Calculate the betweeness centrality for an airline's network for a given year.
    The betweeness is based on the loading (number of flights) on each edge/route.

    Parameters:
    -----------
    year: int
        The year to calculate
    airline: string
        The airline to calculate. Use 'ALL' for entire network.

    Returns:
    --------
    dict :
        Dictionary where keys are nodes and values are their corresponding betweeness

    '''

    sql = '''
        SELECT Origin_Airport_Code AS Origin, Destination_Airport_Code AS Destination,
        COUNT(*) AS Flight_Num
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
        GROUP BY Origin_Airport_Code, Destination_Airport_Code
    '''

    days_in_year = 366 if calendar.isleap(year) == 1 else 365

    params = {'year':year, 'airline':airline,'days_in_year':days_in_year}

    airline_df = db_tools.DBQueries().query_to_df(sql,params=params)

    graph = nx.from_pandas_edgelist(
        airline_df,
        source = 'Origin',
        target = 'Destination',
        edge_attr = 'Flight_Num',
        create_using = nx.DiGraph())

    betweenness_centrality = nx.betweenness_centrality(graph)
    
    return(betweenness_centrality)