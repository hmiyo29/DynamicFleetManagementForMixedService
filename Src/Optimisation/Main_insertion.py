"""
Main file to run the dynamic model with insertion
"""

"""
@author: HMiyoshi
"""
import ast
from geopy.distance import geodesic
import pandas as pd
import numpy as np
from collections import Counter
import copy
import re
import timeit
import time
import sys
import os
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import insertion
# from SETTINGS import Param
# import SETTINGS

# Param = Param(SETTINGS.data_folder, SETTINGS.output_folder)
# # data_folder = SETTINGS.data_folder
# # output_folder = SETTINGS.output_folder
# # scenario = SETTINGS.scenario
# # instance = SETTINGS.inst
# # pattern = SETTINGS.pattern
# def set_param(Param: object, i:int):
#     Param.configs()

    
#     Param.set_param(i)
#     return Param.scenario, Param.inst, Param.pattern

# scenario, instance, pattern = set_param(Param, 80)

# def set_path(scenario: str, inst: int, pattern: list, folder: str):

#     data_path = folder + f"Requests_K0_{pattern[0]}_K1_{pattern[1]}_demand_{scenario}_instance_{inst}.xlsx"

#     return data_path

# data_path = set_path(scenario, instance, pattern, Param.data_folder)


#%% Main function
def main(data_path, output_folder, scenario, instance, pattern):


    global data
    global terminals
    global vessels
    global depots
    global terminals_1
    global terminals_values

    # 
    # read excel
    def read_excel(file_path):
        
        # read the whole excel file
        data = pd.read_excel(file_path, sheet_name=None)
        
        # divide it into different dataframes per sheet
        terminals = (data['N_fred'])
        vessels = (data['K'])
        depots = (data['o'])

        return data, terminals, vessels, depots 

    # read the excel file
    data, terminals, vessels, depots = read_excel(data_path)


    # sparce the terminals coordinates
    terminals_values = terminals.values

    # each terminal location is stored as a string tuple (x, y)
    terminals_1 = [eval(terminals_values[i, 0]) for i in range(len(terminals.values))]
    # add a dummy terminal
    terminals_1.append((0, 0))

    global speed
    global B
    global revisit
    global n_t
    global n_k
    global charge_loc
    global distances
    global empty_distances
    global depots_loc
    global key_R_served
    global key_R_serving
    global key_R_failed
    global dict_R
    global dict_R_assigned
    global dict_R_unassigned
    global dict_R_serving
    global dict_r_k
    global assigned_pool
    global unassigned_pool
    global serving_pool
    global distance_matrix
    global routes
    global depots_loc
    global r_od_index
    global node_list

    depots_loc = []
    for i in range(len(depots)):
        depot = eval(depots['o'].values[i])
        # find which terminal is the depot
        depot_index = terminals_1.index(depot)
        depots_loc.append(depot_index)

    speed = 11.192   # km/h - hard coded
    B = 190    # battery capacity [kWh]
    revisit = 1    # how many times each node can be visited - doesn't matter in the insertion algorithm
    charge_loc = [4, 6]    # same as the initial depots
    # number of terminals
    n_t = len(terminals_1)
    # number of vessels
    n_k = len(vessels)

    node_list = [n_t*n + i for n in range(revisit) for i in range(n_t)]

    # distances
    distances = [0 for _ in range(n_k)]
    empty_distances = [0 for _ in range(n_k)]
# ------------------------------------------------------------------ Initialising ------------------------------------------------------------------
    global depots_temp
    global depots_temp_all

    depots_temp, depots_temp_all = insertion.get_depot_dict(terminals_1, n_k, tt=0, loc=None, revisit=revisit)
  

    # get the time step of the simulation from the sheet names
    sheets = list(data.keys())
    # only the sheets that start with 'R_' are the request data
    sheets = [name for name in sheets if "R_" in name]
    time_steps = [re.findall(r'\d+', name) for name in sheets]
    # get the time step from the sheet names
    time_steps = [int(name[0]) for name in time_steps]

    # Time array for the simulation (6:00-22:00) in minutes
    TT = np.arange(0, 961)

    key_R_served = []
    key_R_failed = []
    serving_pool = []
    assigned_pool = []
    unassigned_pool = []
    sequences = [[] for _ in range(n_k)]
    b_temp = [B for _ in range(n_k)]

    dict_R = {}
    dict_R_assigned = {}
    dict_R_unassigned = {}
    dict_R_serving = {}
    dict_r_k = {}
    routes = [[] for _ in range(n_k)]

    # dataframe to collect the time series of the travel distance and empty distance
    col_k = [f"Travel_distance_k{k}" for k in range(n_k)] + [f"Empty_distance_k{k}" for k in range(n_k)] + [f"loading_pass_{k}" for k in range(n_k)] + [f"loading_parcel_{k}" for k in range(n_k)]
    cols = ["Time", "Total_dist", "Total_empty_dist"] + col_k + ["obj", "Met_Ratio", "N_requests", "Duration"]
    df_distance = pd.DataFrame(columns=cols)

    print("Start the optimisation")
    # %% Run the optimisation

    for tt in TT:
        if tt in time_steps:
            # print(f"Time step {tt}")
            print("Progress: ", str(round(tt/960*100, 2)), "%", end="\r")

            # run the insertion algorithm
            sequences, routes, obj, dict_R, dict_R_assigned, dict_R_unassigned, dict_R_serving, dict_r_k, key_R_served, key_R_failed, depots_temp, depots_temp_all, depots_loc, distances, empty_distances, b_temp, duration, unassigned_pool, assigned_pool, serving_pool = insertion.insertion(time_steps, tt, data, terminals, terminals_1, vessels, key_R_served, key_R_failed, dict_R, dict_R_assigned, dict_R_unassigned, dict_R_serving, unassigned_pool, assigned_pool, serving_pool, dict_r_k, routes, depots_temp, depots_temp_all, depots_loc, charge_loc, revisit, distances, empty_distances, b_temp, n_k, node_list, B, speed)
            
            # print("New sequences and routes")
            # print(sequences)
            # print(routes)
            # print("New objective value")
            # print(obj)

            try:
                met_ratio = len(key_R_served) / (len(key_R_served) + len(key_R_failed))
            except ZeroDivisionError:
                met_ratio = 0

            N_requests = len(dict_R_assigned) + len(dict_R_unassigned) + len(dict_R_serving)

            load_pass = {}
            load_parcel = {}
            for k in range(n_k):
                try:
                    list_route = routes[k][0][6]
                    load_pass[k] = list_route[1]
                    load_parcel[k] = list_route[2]
                # means the vessel has no route
                except IndexError:
                    load_pass[k] = 0
                    load_parcel[k] = 0
            

            row = {"Time": tt, "Total_dist": sum(distances), "Total_empty_dist": sum(empty_distances), **{f"Travel_distance_k{k}": distances[k] for k in range(n_k)}, **{f"Empty_distance_k{k}": empty_distances[k] for k in range(n_k)}, **{f"loading_pass_{k}": load_pass[k] for k in range(n_k)}, **{f"loading_parcel_{k}": load_parcel[k] for k in range(n_k)}, "obj": obj, "Met_Ratio": met_ratio,  "N_requests": N_requests, "Duration": duration}

            df_distance = df_distance._append(row, ignore_index=True)
        
    
    # # write the dataframe to a excel file
    df_distance.to_excel(f"{output_folder}/output_insertion_{scenario}_{pattern}_{instance}.xlsx", index=False)
    print("Optimisation completed!")
    return df_distance



#%% Initial step function

def ini_step(instance, data, tt=0):
   # make the initial variables global

    # global distances
    
    assigned_pool = []
    unassigned_pool = []
    serving_pool = []
    key_R_failed = []
    key_R_served = []
    key_R_serving = []
    dict_R = {}
    dict_R_assigned = {}
    dict_R_unassigned = {}
    dict_R_serving = {}
    dict_r_k = {}

    node_list = [n_t*n + i for n in range(revisit) for i in range(n_t)]

    # 1. Read the new request data
    R = data[f'R_{tt}']
    # add 'key' column to the dataframe
    R['key'] = [100*tt+r for r in range(len(R))]

    # print("here1")

    # 2. add the requests to the dictionary
    for r in range(len(R)):
        dict_R[100*tt+r] = R.loc[r]
    
    r_od_index = get_o_d_index(dict_R, terminals)
    # add attributes
    R = get_pickup(R)
    R = get_service_time(R)

    # get the distance matrix
    distance_matrix = get_distance_matrix(terminals)

    # tile the distance matrix for multiple visits - max 5 times
    distmat_all = np.tile(distance_matrix, (revisit, revisit))
    
    travel_time = distance_matrix / speed *60    # in minutes
    travel_time_all = distmat_all / speed *60    # in minutes

    for r in range(len(R)):
        dict_R[100*tt+r] = R.loc[r]
    
    for r in range(len(R)):
        unassigned_pool.append(100*tt+r)
    
    R_unassigned = pd.DataFrame([dict_R[i] for i in unassigned_pool])
    dict_R_unassigned = {i: dict_R[i] for i in unassigned_pool}
    
    
    # Set the depots of the vessels
    depots_loc = []
    for i in range(len(depots)):
        depot = eval(depots['o'].values[i])
        # find which terminal is the depot
        depot_index = terminals_1.index(depot)
        depots_loc.append(depot_index)

    # Check the feasibility of requests
    R_unassigned_feas = R_unassigned.copy()
    # in the unassigned pool, find requests that are not feasible to be served at this timestep

    # if no vessel is within the maximum waiting time of the request, then the request is not feasible
    # use the current location of the vessels to check if the request is feasible
    # iterate through the unassigned requests
    for r_1 in R_unassigned_feas.index:
        # print(f"Checking the feasibility of request {R_unassigned_feas.at[r_1, 'key']}")
        pickupend = R_unassigned_feas.at[r_1,'ptw']
        service_time = R_unassigned_feas.at[r_1, 'st']
        threshold = pickupend - tt - service_time
        # consider the vessel type for available vessels
        available_vessels = [k for k in range(n_k) if (vessels.at[k, "type"] == R_unassigned_feas.at[r_1, 's']) or (vessels.at[k, "type"] == 2)]
        for k in available_vessels.copy():
            # get the distance between the current location of the vessel and the origin of the request

            tt_cur_o = travel_time[depots_loc[k], r_od_index[R_unassigned_feas.at[r_1, 'key']][0]]

            # if the vessel is not within the maximum waiting time of the request, then the request is not feasible
            if (tt_cur_o > threshold):
                available_vessels.remove(k)
            else:
                pass

        if len(available_vessels) == 0:    
            # drop the request from the unassigned pool
            key_R_failed.append(R_unassigned.at[r_1, 'key'])
            R_unassigned_feas = R_unassigned_feas.drop(r_1)

            # remove the request from the dictionary
            del dict_R_unassigned[R_unassigned.at[r_1, 'key']]
            # print("\t", f"Request {R_unassigned.at[r_1, 'key']} is not feasible to be served at time step {tt}")

        else:
            # print("\t", f"Request {R_unassigned.at[r_1, 'key']} is feasible to be served at time step {tt}")
            pass
    R_unassigned = R_unassigned_feas.copy()


    obj, x_sol, y_sol, z_sol, t_sol, t_s_sol, t_e_sol, t_n_sol, t_r_sol, t_rs_sol, t_re_sol, t_rn_sol, b_sol, b_d_sol, b_c_sol, zeta_q_sol, duration, removed_keys = run_inimodel(n_k, n_t, dict_R_assigned={}, dict_R_unassigned=dict_R_unassigned, dict_R_serving={}, dict_r_k={}, dist_mat=distmat_all, depots_loc=depots_loc, charge_loc=depots_loc, vessels=vessels, revisit=revisit, tt=tt, multi=True)


    print("Calculation time: ", duration)
    
    # from the unassigned pool, find the requests that are assigned to the vessels from the solution y_sol
    for key, value in y_sol.items():
        if key[1] in unassigned_pool:
            assigned_pool.append(key[1])
            unassigned_pool.remove(key[1])

    # create a dictionary which stores the vessel that serves the request
    dict_r_k = {}
    for key, value in y_sol.items():
        if value == 1:
            dict_r_k[key[1]] = key[0]

    # update the dictionary of requests
    dict_R_assigned = {i: dict_R[i] for i in assigned_pool}
    dict_R_unassigned = {i: dict_R[i] for i in unassigned_pool}

    # create dataframe 
    R_assigned = pd.DataFrame([item for item in dict_R_assigned.values()])
    R_unassigned = pd.DataFrame([item for item in dict_R_unassigned.values()])


    # get the loading matrix
    y_values = y_sol.copy()
    loading_matrix, loading_pass, loading_parcel = get_loading(vessels, node_list, dict_R_assigned, y_values)

    # get the routes of the vessels
    routes = []

    for k in range(n_k):
        # i, j, arrival time at i, arrival time at j, loading, serving request ids
        mat_route = np.zeros((1, 6))
        for i in node_list:
            for j in node_list:
                if x_sol[(k, i, j)] > 0:
                    # obtain the serving requests
                    serving_requests = []
                    for key in dict_R_assigned.keys():
                        if y_values[(k, key, i, j)] > 0:
                            # adding the key of the request to the serving requests
                            serving_requests.append(key)
                    mat_route = np.vstack((mat_route, [i, j, round(t_e_sol[(k, i)], 3), round(t_sol[(k, j)], 3), str([loading_matrix[k, i, j], loading_pass[k, i, j], loading_parcel[k, i, j]]), str(serving_requests)]))

        # remove the first row
        mat_route = np.delete(mat_route, 0, 0)
        # sort the route based on the arrival time at the nodes
        mat_route = mat_route[np.argsort(mat_route[:, 2].astype(float))]

        routes.append(mat_route)

    dict_R_assigned_temp = copy.deepcopy(dict_R_assigned)
    for key in dict_R_assigned_temp.keys():
        for k, route in enumerate(routes):
            # print(route)
            # print(f"Searching the final trip of request {key} by vessel {k}")
            serving = route[:, 5]
            # print(serving)
            # Convert the string to a list
            # try:
            serving_list = [ast.literal_eval(serving[i]) for i in range(len(serving))]
            # print(serving_list)

            # get the row of the route if the key is in the serving list
            ind = [i for i in range(len(serving_list)) if key in serving_list[i]]
            # get the row of the route where the request arrive at the destination
            try:
                serve_route_first = route[ind[0]]
                serve_route_last = route[ind[-1]]

                # compare the current time step and the arrival time at the destination and remove the request from the assigned pool if the vessel has arrived. The request is stored in the R_served pool
                if float(serve_route_last[3]) <= tt:
                    # print("\t", f"Request {key} is completed serving by vessel {k}")
                    
                    key_R_served.append(key)
                    R_assigned = R_assigned[R_assigned['key'] != key]
                    assigned_pool.remove(key)
                    key_R_serving.remove(key)
                    # remove the request from the dictionary
                    del dict_R_assigned[key]

                elif (float(serve_route_last[3]) > tt) & (float(serve_route_first[2]) <= tt):
                    # print("\t", f"Request {key} is currently being served by vessel {k}")
                    key_R_serving.append(key)
                    # remove the request from the dictionary
                    del dict_R_assigned[key]

                else:
                    # print("\t", f"Request {key} is not started served  by vessel {k} yet")
                    pass
                
            except IndexError:
                # print("\t", f"No route found for request {key} by vessel {k}")
                pass
            
    R_serving = pd.DataFrame([dict_R[i] for i in key_R_serving])
    dict_R_serving = {i: dict_R[i] for i in key_R_serving}

    
    return dict_R, dict_R_assigned, dict_R_unassigned, dict_R_serving, dict_r_k, obj, x_sol, y_sol, z_sol, t_sol, t_s_sol, t_e_sol, t_n_sol, t_r_sol, t_rs_sol, t_re_sol, t_rn_sol, b_sol, b_d_sol, b_c_sol, zeta_q_sol, routes, terminals_1, depots_loc, r_od_index, duration, key_R_failed



#%% Other functions

# read excel
def read_excel(file_path):
    
    # read the whole excel file
    data = pd.read_excel(file_path, sheet_name=None)
    
    # divide it into different dataframes per sheet
    terminals = (data['N_fred'])
    vessels = (data['K'])
    depots = (data['o'])

    return data, terminals, vessels, depots 



def get_distance(origin, destination):
    """"
    Calculate the distance between two points
    input:
    origin: string, "(x, y)"
    destination: string, "(x, y)"

    output:
    distance: float, the distance between the two points
    """
    # convert the string to tuple
    origin = ast.literal_eval(origin)
    destination = ast.literal_eval(destination)

    # calculate the geographical distance
    distance = geodesic(origin, destination).kilometers

    return distance

def get_distance_matrix(terminals):
    """
    Calculate the distance matrix between all terminal locations
    """

    # get the number of terminals (excluding the dummy terminal)
    n = len(terminals)

    # create an empty distance matrix
    distance_matrix = np.zeros((n, n))

    # fill the distance matrix
    for i in range(n):
        for j in range(n):
            distance_matrix[i, j] = get_distance(terminals['N'][i], terminals['N'][j])
    
    # stack the dummy terminal to the distance matrix with zeros
    distance_matrix = np.vstack((distance_matrix, np.zeros(n)))
    distance_matrix = np.hstack((distance_matrix, np.zeros((n+1, 1))))

    return distance_matrix

def get_temp_distmat(dist_mat, n_k, tt, locs, revisit):
    """
    Get the distance matrix including the current locations of vessels and terminals
    Input:
    dist_mat: np.array, distance matrix between all terminal locations
    n_k: int, the number of vessels
    tt: the time step
    locs: list of the current locations of the vessels
    revisit: int, the number of times the vessels can revisit the terminals


    Output:
    temp_distmat: np.array, distance matrix including the current locations of vessels and terminals
    """

    
    # make a list of coordinates of terminals and vessels
    term_coords = [terminals['N'][i] for i in range(len(terminals))]
    ves_coords = [str(loc[1:]) for loc in locs]

    # calculate the distance between all terminals and vessels
    temp_distmat = np.zeros((len(term_coords) + len(ves_coords) + 1, len(term_coords) + len(ves_coords) + 1))
    temp_distmat[:len(term_coords)+1, :len(term_coords)+1] = dist_mat
    
    for i in range(len(term_coords)):
        for j in range(len(ves_coords)):
            dist = get_distance(term_coords[i], ves_coords[j])
            temp_distmat[i, j+len(term_coords)+1] = dist
            temp_distmat[j+len(term_coords)+1, i] = dist
    for i in range(len(ves_coords)):
        for j in range(len(ves_coords)):
            # distance
            dist = get_distance(ves_coords[i], ves_coords[j])
            temp_distmat[i+len(term_coords)+1, j+len(term_coords)+1] = dist
    
    # for the distance from dummy to vessels - large number
    temp_distmat[len(term_coords), len(term_coords):] = 10000

    temp_distmat_all = np.tile(temp_distmat, (revisit, revisit))


    return temp_distmat_all


def get_location(tt, n_k, terminals_1, node_list, routes, depots_temp, depots_loc, distances, empty_distances):
    """
    Get the location of the vessels at each time
    input: 
    tt: float, the current time
    n_k: int, the number of vessels
    terminals_1: list, the terminal locations
    routes: list, the routes of the vessels
    depots_temp: dict, the coordinates of the terminals
    depots_loc: list, the location indices of the vessels
    distances: list, the distance travelled by the vessels
    empty_distances: list, the distance travelled by the vessels without the request


    output: 
    current_location: list of the location of the vessels at the time, [vessel, x, y]
    distances: list, the distance travelled by the vessels
    empty_distances: list, the distance travelled by the vessels without the request
    """
    
    n_t = len(terminals_1)
    end_depot = n_t - 1

    current_location = [(k, 0, 0) for k in range(n_k)]
    distances_copy = distances.copy()
    empty_distances_copy = empty_distances.copy()
    # take routes as input instead
    for k in range(n_k):
        distance = distances_copy[k]
        empty_distance = empty_distances_copy[k]
        # get the route of vessel k
        route = routes[k]
        # remove the row where the destination is the dummy terminal
        route = [r for r in route if r[1] != str(end_depot)]
        try:
            # search for the row that is currently being served
            if tt <= float(route[0][2]):
                # vessel is at the current location
                location = int(route[0][0])
                current_location[k] = (k, depots_temp[location][0], depots_temp[location][1])

            elif tt >= float(route[-1][3]):
                # vessel is at the last location
                location = int(route[-1][1])
                current_location[k] = (k, depots_temp[location][0], depots_temp[location][1])

                # loop over all rows in route to calculate the distance travelled
                for r in route:
                    ori = int(r[0])
                    des = int(r[1])
                    distance += get_distance(str(depots_temp[ori]), str(depots_temp[des]))
                    if r[5] == '[]':
                        empty_distance += get_distance(str(depots_temp[ori]), str(depots_temp[des]))
                
                
            
            else:
                # find the row that is currently being served
                try:
                    # print("route in service")
                    arr_trip = [r for r in route if float(r[2]) <= tt <= float(r[3])]
                    trip = [r for r in route if float(r[2]) <= tt <= float(r[3])][0]
                    # print("route in the middle of a trip")
                
                    ori = int(trip[0])
                    des = int(trip[1])
                    t_o = float(trip[2])
                    t_d = float(trip[3])

                    # coordinates of the origin and destination
                    o = depots_temp_all[ori]
                    d = depots_temp_all[des]

                    # get the current location - linear interpolation
                    x = o[0] + (d[0] - o[0]) * (tt - t_o) / (t_d - t_o)
                    y = o[1] + (d[1] - o[1]) * (tt - t_o) / (t_d - t_o)

                    current_location[k] = (k, x, y)

                    # calculate the distance travelled
                    for ind in range(len(route)):
                        if float(route[ind][3]) <= tt:
                            
                            ori_ind = int(route[ind][0])
                            des_ind = int(route[ind][1])
                            distance += get_distance(str(depots_temp[ori_ind]), str(depots_temp[des_ind]))
                            if route[ind][5] == '[]':
                                empty_distance += get_distance(str(depots_temp[ori_ind]), str(depots_temp[des_ind]))

                    # calculate the distance travelled in the current trip
                    distance += get_distance(str(o), str(current_location[k][1:]))
                    if trip[6] == [0, 0, 0]:
                        empty_distance += get_distance(str(o), str(current_location[k][1:]))

                    pass               


                except:
                    # vessel is dwelling at one of the terminals
                    cur_terminal = depots_loc[k]
                    ori_terminal = depots_loc[k]

                    # print("vessel is dwelling at the terminal")
                    for r in route:
                        if float(r[3]) <= tt:
                            cur_terminal = int(r[1])
                            ori_terminal = int(r[0])
                            distance += get_distance(str(depots_temp[ori_terminal]), str(depots_temp[cur_terminal]))
                            if trip[6] == [0, 0, 0]:
                                empty_distance += get_distance(str(depots_temp[ori_terminal]), str(depots_temp[cur_terminal]))
                        
                        else:
                            break
                    current_location[k] = (k, depots_temp[cur_terminal][0], depots_temp[cur_terminal][1])
                            

        except IndexError:
            # vessel has no route plan
            # print("no route found")
            location = depots_loc[k]
            current_location[k] = (k, depots_temp[location][0], depots_temp[location][1])
        
        distances[k] = distance
        empty_distances[k] = empty_distance
        

    return current_location, distances, empty_distances


def get_o_d_index(dict_R, terminals):
    """
    Get the index of the origin and destination of each request
    """

    # create an empty list to store the origin and destination index
    o_d_index = {}

    # fill the list
    for key, values in dict_R.items():
        # print(key, values)
        origin = values['p']
        destination = values['d']
        # print(destination)

        # find the index of the origin and destination (dataframe index)
        origin_index = terminals['N'][terminals['N'] == origin].index.values[0]
        destination_index = terminals['N'][terminals['N'] == destination].index.values[0]

        o_d_index[key] = (origin_index, destination_index)

    return o_d_index


def get_arcs(nodes):
    """
    Get the arcs between all nodes locations
    """
    # get the number of terminals
    n = len(nodes)

    # create an empty list to store the arcs
    arcs = []

    # fill the list
    for i in range(n):
        for j in range(n):
            if i != j:
                arcs.append((i, j))

    return arcs

# service time calculation
# service time per request is linear to its size
def get_service_time(R, coeff_p=0.25, coeff_f=0.5) -> pd.DataFrame:
    """
    Calculate the service time for each request
    input:
    R: dataframe, the request dataframe
    coeff_p: float, the coefficient for the serice time for passengers (default 0.25)
    coeff_f: float, the coefficient for the service time for freight (default 0.5)

    output:
    R: dataframe, the request dataframe with the service time added
    """
    # get the number of requests
    n = len(R)

    # add a new column "st"
    R['st'] = 0

    # fill the list
    for r in range(n):
        size = R.loc[r, 'qr']
        r_type = R.loc[r, 's']
        time = coeff_p * r_type * size + coeff_f * (1-r_type) * size
        # add as a new column to the dataframe
        R.loc[r, 'st'] = time

    return R

# add pickup end time window
def get_pickup(R, max_wait=15) -> pd.DataFrame:
    """
    Add the pickup end time window for each request
    input:
    R: dataframe, the request dataframe
    max_wait: float, the maximum waiting time for the passengers (default 15 minutes)

    output:
    R: dataframe, the request dataframe with the pickup end time window added
    """
    # get the number of requests
    n = len(R)

    # add a new column "ptw"
    R['ptw'] = 0

    # fill the list
    for r in range(n):
        if R.loc[r, "s"] == 0:
            R.loc[r, 'ptw'] = 960
        # add as a new column to the dataframe
        elif R.loc[r, "s"] == 1:
            R.loc[r, 'ptw'] = R.loc[r, 'a'] + max_wait

    return R

def get_loading(vessels, node_list, dict_R, y_values):
    """
    Get the loading of each vessel at each terminal
    input: vessel, node_list, request dataframe, y_values
    output: loading matrix (loading of each vessel when travelling from i to j)
    """

    # calculate the loading of each vessel
    loading_matrix = np.zeros((len(vessels), len(node_list), len(node_list)), dtype=float)

    loading_pass = np.zeros((len(vessels), len(node_list), len(node_list)), dtype=float)

    loading_parcel = np.zeros((len(vessels), len(node_list), len(node_list)), dtype=float)

    # get the key of each request
    keys = dict_R.keys()

    for k in range(len(vessels)):
        for i in range(len(node_list)):
            for j in range(len(node_list)):
                # loading of each vessel when serving from i to j
                loading = 0
                loading_p = 0
                loading_parc = 0
                for key in keys:
                    # if the request r is served from i to j by vessel k
                    try:
                        if y_values[(k, key, i, j)] > 0:
                            loading += dict_R[key]['qr']
                            if dict_R[key]['s'] == 0:
                                loading_parc += dict_R[key]['qr']
                            elif dict_R[key]['s'] == 1:
                                loading_p += dict_R[key]['qr']
                        else:
                            pass
                    except KeyError:
                        pass
                loading_matrix[k, i, j] = loading
                loading_pass[k, i, j] = loading_p
                loading_parcel[k, i, j] = loading_parc
    
    return loading_matrix, loading_pass, loading_parcel


def get_maxtransit_time(R, travel_time, r_od_index, factor=2):
    """
    Get the maximum transit time for each request
    """
    # maximum transit time is dependent on the travel time between the pickup and delivery locations
    gamma = travel_time * factor

    for r in range(len(R)):
        o, d = r_od_index[r]
        if R.loc[r, 's'] == 1:
            newgamma = gamma[o, d]
        else:
            newgamma = 960
        
        R.loc[r, 'gamma'] = newgamma
    
    return R

    
def get_depot_dict(terminals_1, n_k, tt, loc, revisit):
    """
    Create a dictionary to store the terminals and vessels locations indices and coordinates
    """
    # terminals
    depot_dict = {}
    for i in range(len(terminals_1)):
        depot_dict[i] = terminals_1[i]

    # vessels
    if loc != None:
        for k in range(n_k):
            depot_dict[len(terminals_1)+k] = loc[k][1:]
    else:
        pass

    # duplicate the dictionary ten times
    # depot_dict[0] = depot_dict[11] = depot_dict[22] = depot_dict[33] = ...
    depot_dict_all = {}
    for n in range(len(depot_dict)):
        for i in range(revisit):
            depot_dict_all[i*len(depot_dict) + n] = depot_dict[n]


    return depot_dict, depot_dict_all

# %% Run main

if __name__ == "__main__":
    main(data_path, SETTINGS.output_folder)
