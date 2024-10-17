"""
Main file to create request dataset for the simulation model
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the classes
from Class_Parcel import Par_requests
from Class_Pass import Pass_requests


# number of instances
N_inst = 2
# folder to read the data
folder = '../../data/'

# folder to save the data
folder_data = folder + 'data_/'

# create a folder to save the data if it does not exist
if not os.path.exists(folder_data):
    os.makedirs(folder_data)

#%% main
def main(N_inst: int, folder: str, folder_data: str):
    """
    Main function to generate the requests datasets
    """
    # vessel purposes possibility
    # 0: passenger, 1: parcel, 2: mixed
    patterns = [[2,2], [1,2], [2,1], [0,2], [2,0], [0,1], [1,0]]

    # demand level - in the parameter files, the demand level is set to 'high' or 'low'
    demand = ['low', 'high']

    # Collect the statistics of the requests per instance and demand level
    df_allstats = pd.DataFrame(columns=['Instance', 'Demand', 'Total_requests', 'Passenger_requests', 'Parcel_requests'])


    for scenario in demand:    
        for instance in range(N_inst):

            # For each instance and demand level, the temporal distribution of the requests is created

            print(f'Instance {instance} and demand level {scenario}')

            # create the requests
            L_requests, L_pass, L_par, L_requests_tt, L_pass_tt, L_par_tt = request_generation(folder, folder_data, patterns, instance, scenario)

            df_request_dist = pd.DataFrame(columns=['Time', 'Time_h',  'Total_requests', 'Passenger_requests', 'Parcel_requests'])

            # the time horizon is defined as 16 hours (960 minutes)
            for key in range(961):
                
                if key not in L_requests_tt.keys():
                    L_requests_tt[key] = 0
                    
                if key not in L_pass_tt.keys():
                    L_pass_tt[key] = 0
                
                if key not in L_par_tt.keys():
                    L_par_tt[key] = 0

                time = pd.to_datetime(key, unit='m') + pd.Timedelta(hours=6)    # 6:00 is the starting time

                # only need the time, not the date
                time = time.time()
                # create a row for the statistics
                row_dist = {'Time': key, 'Time_h': time, 'Total_requests': L_requests_tt[key], 'Passenger_requests': L_pass_tt[key], 'Parcel_requests': L_par_tt[key]}
                df_request_dist = df_request_dist._append(row_dist, ignore_index=True)

            # save the request distribution to an excel file
            df_request_dist.to_excel(folder_data + f'Request_distribution_instance_{instance}_demand_{scenario}.xlsx', index=False)

            

            # create a row for the statistics
            row = {'Instance': instance, 'Demand': scenario, 'Total_requests': L_requests, 'Passenger_requests': L_pass, 'Parcel_requests': L_par}
            
            df_allstats = df_allstats._append(row, ignore_index=True)

    # save the statistics to an excel file
    df_allstats.to_excel(folder_data + '/Statistics_requests.xlsx', index=False)


#%% Single instance generation
def request_generation(folder: str, folder_data: str, patterns: list, instance: int , high: str) -> tuple:
    """
    Function to create the requests, single instance
    """
    # parameters for the request generation
    # file_path = folder + f'Para_K0_{p[0]}_K1_{p[1]}_' + high + '.xlsx'
    file_path = folder + f'Para_' + high + '.xlsx'

    # create the parcel requests
    parcel_requests = Par_requests(file_path)
    parcel_requests.all_process()

    # create the passenger requests
    passenger_requests = Pass_requests(file_path)
    passenger_requests.all_process()

    # combine the requests
    dict_requests_all = combine_requests(passenger_requests, parcel_requests)

    # Collect the statistics of the requests --- length of dictionary is not the same as the number of requests FIX 
    L_pass = len(passenger_requests.dict_requests)    # Total number of passenger requests
    L_par = len(parcel_requests.dict_requests)    # Total number of parcel requests

    L_requests = L_par + L_pass   # Total number of requests

    # collect the number of requests per timestep
    L_requests_tt = {}
    for key in dict_requests_all.keys():
        L_requests_tt[key] = len(dict_requests_all[key])
    
    # collect the number of passenger requests per timestep
    L_pass_tt = {}
    for key in passenger_requests.dict_requests_tt.keys():
        L_pass_tt[key] = len(passenger_requests.dict_requests_tt[key])
    
    # collect the number of parcel requests per timestep
    L_par_tt = {}
    for key in parcel_requests.dict_requests_tt.keys():
        L_par_tt[key] = len(parcel_requests.dict_requests_tt[key])

    # create a folder to save the data
    if not os.path.exists(folder_data):
        os.makedirs(folder_data)
    
    for p in patterns:
        # define the file path
        file_path = folder + f'Para_' + high + '.xlsx'
        # define the filename
        filename = folder_data + f'Requests_K0_{p[0]}_K1_{p[1]}_demand_' + high + f'_instance_{instance}.xlsx'
        # write the requests to an excel file
        write_excel(filename, file_path, dict_requests_all, p)

        print('Requests are created and saved to the excel file')

    return L_requests, L_pass, L_par, L_requests_tt, L_pass_tt, L_par_tt


# %% Other functions

def combine_requests(Passenger_requests: object, Parcel_requests:object) -> dict:

    """
    Combine the requests of passengers and parcels
    """

    keys_pass = list(Passenger_requests.dict_requests_tt.keys())
    keys_par = list(Parcel_requests.dict_requests_tt.keys())

    dict_requests_all = {}
    for tt in range(Passenger_requests.total_time):
        if (tt in keys_pass) and (tt in keys_par):
            df_pass = Passenger_requests.dict_requests_tt[tt]
            df_par = Parcel_requests.dict_requests_tt[tt]
            df_all = pd.concat([df_pass, df_par], axis=0, ignore_index=True)
            dict_requests_all[tt] = df_all
        elif (tt in keys_pass) and (tt not in keys_par):
            dict_requests_all[tt] = Passenger_requests.dict_requests_tt[tt]
        elif (tt not in keys_pass) and (tt in keys_par):
            dict_requests_all[tt] = Parcel_requests.dict_requests_tt[tt]
        else:
            pass

    return dict_requests_all


def write_excel(filename, file_path, dict_requests_all, pattern):
    # write the dataframe to excel sheet
    # create a sheet in the excel file
    writer = pd.ExcelWriter(filename, engine='openpyxl')

    data = pd.read_excel(file_path, sheet_name=None)
    terminals = (data['N_fred'])
    vessels = (data['K'])
    for i in range(len(pattern)):
        vessels.at[i, 'type'] = pattern[i]
    
    depots = (data['o'])
    # first sheet is N_fred
    terminals.to_excel(writer, sheet_name='N_fred', index=False)
    # second sheet is K
    vessels.to_excel(writer, sheet_name='K', index=False)
    # third sheet is o
    depots.to_excel(writer, sheet_name='o', index=False)

    # create a sheet for the requests
    for key in dict_requests_all.keys():
        # print(key)
        dict_requests_all[key].to_excel(writer, sheet_name='R_'+str(key), index=False)

    writer.close()


#%% Run main
if __name__ == '__main__':
    # run the main function
    main(N_inst, folder, folder_data)
    print('All requests are created and saved to the excel files')