"""
Insertion heuristic for route plan optimisation 

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
from itertools import product


def insertion(time_steps, tt, data, terminals, terminals_1, vessels, key_R_served, key_R_failed, dict_R, dict_R_assigned, dict_R_unassigned, dict_R_serving, unassigned_pool, assigned_pool, serving_pool, dict_r_k, routes, depots_temp, depots_temp_all, depots_loc, charge_loc, revisit, distances, empty_distances, b_temp, n_k, node_list, B, speed):
    """"
    Insertion heuristic for route plan optimisation
    """

    # load new requests
    try:
        R = data[f'R_{tt}']
        # add pickup end time window
        R = get_pickup(R)
        # add service time
        R = get_service_time(R)
        # add 'key' column to the dataframe
        R['key'] = [100*tt+r for r in range(len(R))]
        # 2. add the requests to the dictionary
        for r in range(len(R)):
            dict_R[100*tt+r] = R.loc[r]    
            # add the new requests to the unassigned pool
            unassigned_pool.append(100*tt+r)
            dict_R_unassigned[100*tt+r] = R.loc[r]
    except KeyError:
        # print(f"No new requests at time step {tt}")
        print(" ")
    
    # update the dataframes and the dictionaries for the requests
    R_assigned = pd.DataFrame([dict_R[i] for i in assigned_pool])
    R_unassigned = pd.DataFrame([dict_R[i] for i in unassigned_pool])
    R_serving = pd.DataFrame([dict_R[i] for i in serving_pool])

    dict_R_assigned = {i: dict_R[i] for i in assigned_pool}
    dict_R_unassigned = {i: dict_R[i] for i in unassigned_pool}
    dict_R_serving = {i: dict_R[i] for i in serving_pool}

    # update the current locations of the vessels
    loc, distances, empty_distances = get_location(tt, n_k, terminals_1, node_list, routes, depots_temp, depots_loc, distances, empty_distances)

    # remove the infeasible requests
    dict_R_unassigned_copy = copy.deepcopy(dict_R_unassigned)
    for key, item in dict_R_unassigned_copy.items():
        if item['ptw'] < tt:
            key_R_failed.append(key)
            unassigned_pool.remove(key)
            R_unassigned = R_unassigned[R_unassigned['key'] != key]
            del dict_R_unassigned[key]

    # keep the old depots and locations
    depots_temp_old = copy.deepcopy(depots_temp)
    depot_temp_all_old = copy.deepcopy(depots_temp_all)

    # get the new depot dictionary
    depots_temp, depots_temp_all = get_depot_dict(terminals_1, n_k, tt, loc, revisit)

    
    # rest of the time steps
    if tt == time_steps[0]:
        depots_loc = depots_loc
    # the first time step
    else:
        depots_loc = list(depots_temp.keys())[len(terminals_1):]


    distance_matrix = get_distance_matrix(terminals)

    # update the distance matrix and the travel time matrix
    dist_temp = get_temp_distmat(distance_matrix, n_k, tt, locs=loc, revisit=revisit, terminals=terminals)
    travel_time_temp = dist_temp / speed * 60    # minutes

    # get the index of the origin and destination of each request - {r_key: (origin_index, destination_index)}
    r_od_ind_assigned = get_o_d_index(dict_R_assigned, terminals)
    r_od_ind_unassigned = get_o_d_index(dict_R_unassigned, terminals)
    r_od_ind_serving = get_o_d_index(dict_R_serving, terminals)

    r_od_ind = {**r_od_ind_assigned, **r_od_ind_unassigned, **r_od_ind_serving}

    depots_loc_real = copy.deepcopy(depots_loc)

    # check if the depots are the same as one of the terminals
    end_depot = len(terminals_1) - 1

    for k in range(n_k):
        depot_loc_k = depots_loc[k]
        for i, depot in depots_temp.items():

            # if the depot is the same as the terminal
            if (depot[0] == depots_temp[depot_loc_k][0]) and (depot[1] == depots_temp[depot_loc_k][1]):

                depots_loc_real[k] = i
                break
            elif i == end_depot:
                break
            else:
                continue
    # print("Preprocessing done")
    # routes is now a array of [i, j, t_d_i, t_j, b_d_i, b_j, loading, serving_requests] - make sure to map the variables correctly
    # now it also does not include the dummy terminal

# %% Status updates
    route_updated = copy.deepcopy(routes)
    sequences = [[], []]

    for k in range(n_k):
        route = routes[k]

        # check all keys in the route[:,-1]
        requests = [r[-1] for r in route]
        flatten = lambda l: [item for sublist in l for item in sublist]
        # all the requests in the route of the vessel
        unique_requests = list(set(flatten(requests)))


        # when the vessel had no route
        if len(route) == 0:
            sequences[k] = [depots_loc[k]]
            b_k_temp = b_temp[k]
            pass

        else:
            # vessel is at the origin of the first trip - should not happen in the insertion heuristic
            if tt < float(route[0][2]):
                raise ValueError("Assumption violated, route plan should start immediately in the insertion heuristic")
                sequences[k] = seq_from_route(route)
                route[0, 2] = tt
                b_k_temp = b_temp[k]


            # vessel is at the destination of the last trip
            elif tt >= float(route[-1][3]):
                # print("Vessel is at the destination of the last trip")
                i = int(route[-1][1])
                b_end = route[-1][5]
                t_end = route[-1][3]

                # charging linear to time
                if i in charge_loc:
                    b_k_temp = min(b_end + 100/60*(tt - t_end), B)
                # stays the same as arrival to the destination
                else:
                    b_k_temp = b_end

                # all requests are served
                # every request in unique_requests is served
                for req in unique_requests:
                    # remove the request from either assigned or serving pool
                    if req in unassigned_pool:
                        unassigned_pool.remove(req)
                        del dict_R_unassigned[req]
                        

                    if req in assigned_pool:
                        assigned_pool.remove(req)
                        del dict_R_assigned[req]
                        
                    
                    if req in serving_pool:
                        serving_pool.remove(req)
                        del dict_R_serving[req]
                        
                    
                    # add the request to the served pool
                    if req not in key_R_served:
                        key_R_served.append(req)

                # all trips are completed
                route_updated[k] = []
                sequences[k] = [depots_loc[k]]

                # print("Complete last trip update")

            # vessel is in the middle of the route
            else:
                try:
                    # vessel is in the middle of a trip
                    row = [r for r in route if float(r[2]) <= tt <= float(r[3])][0]
                    row_arr = [r for r in route if float(r[2]) <= tt <= float(r[3])]
                    # print("Vessel is in the middle of a trip")
                    row_copy = copy.deepcopy(row)
                    ind = np.where(np.all(route == row, axis=1))[0][0]

                    dist = get_distance(str(depots_temp[int(row[0])]), str(loc[k][1:]))     # km


                    # battery level at the beginning of the trip
                    b_start = float(row[4])
                    b_k_temp = b_start - dist * 1

                    # request status check
                    # first the serving requests
                    serving_requests = row[-1]
                    for req in serving_requests:
                        # if the request is served
                        if req in assigned_pool:
                            assigned_pool.remove(req)
                            del dict_R_assigned[req]
                        if req in unassigned_pool:
                            unassigned_pool.remove(req)
                            del dict_R_unassigned[req]
                        
                        # add the request to the serving pool
                        if req not in serving_pool:
                            serving_pool.append(req)
                            dict_R_serving[req] = dict_R[req]
                        
                    # the requests that appears in the route after the row is the assigned requests
                    rest_route = route[ind+1:]

                    assigned_requests = list(set(flatten([r[-1] for r in rest_route])))

                    # get the requests from the unique_requests that are not in the assigned_requests - completed serving requests
                    served_requests = [req for req in unique_requests if req not in assigned_requests and req not in serving_requests]
                    for req in served_requests:
                        # remove the request from the assigned or serving pool
                        if req in assigned_pool:
                            assigned_pool.remove(req)
                            del dict_R_assigned[req]
                        if req in serving_pool:
                            serving_pool.remove(req)
                            del dict_R_serving[req]
                        if req in unassigned_pool:
                            unassigned_pool.remove(req)
                            del dict_R_unassigned[req]
                        # add the request to the served pool
                        if req not in key_R_served:
                            key_R_served.append(req)
                    
                    # build the current trip
                    row_copy[0] = depots_loc[k]
                    row_copy[2] = float(tt)
                    row_copy[4] = b_k_temp

                    # remove the trips that are completed and stack the row_copy as the first row
                    route_updated[k] = np.vstack((row_copy, rest_route))

                    # get the sequence of the vessel
                    sequences[k] = seq_from_route(route_updated[k])



                # vessel is dwelling at the terminal
                except:
                    cur_terminal = depots_loc[k]
                    for i in range(len(route)):
                        for j in range(2, 4):
                            criteria = (route[i][j] < tt)
                            if criteria:
                                cur_terminal = int(route[i][j-2])
                                b_now = float(route[i][j+2])
                                t_now = float(route[i][j])
                                continue
                            if criteria == False:
                                last_done_trip = i-1
                                break
                        if criteria == False:
                            break

                    if cur_terminal in charge_loc:
                        b_k_temp = min(b_now + 100/60*(tt - float(t_now)), B)
                    else:
                        b_k_temp = b_now
                    
                    done_route = route[:last_done_trip+1]
                    rest_route = route[last_done_trip+1:]

                    # get the unique requests in both route respectively
                    done_requests = list(set(flatten([(r[-1]) for r in done_route])))
                    rest_requests = list(set(flatten([(r[-1]) for r in rest_route])))

                    for req in unique_requests:
                        # for all unique requests in the route, if only in done requests, remove from assigned or serving pool and add to served pool  
                        if req in done_requests and req not in rest_requests:
                            if req in assigned_pool:
                                assigned_pool.remove(req)
                                del dict_R_assigned[req]
                            if req in unassigned_pool:
                                unassigned_pool.remove(req)
                                del dict_R_unassigned[req]
                            if req in serving_pool:
                                serving_pool.remove(req)
                                del dict_R_serving[req]
                            if req not in key_R_served:
                                key_R_served.append(req)
                        # if req is in both, remove from assigned pool and add to serving pool
                        elif (req in done_requests) and (req in rest_requests):
                            if req in assigned_pool:
                                assigned_pool.remove(req)
                                del dict_R_assigned[req]
                            if req not in serving_pool:
                                serving_pool.append(req)
                                dict_R_serving[req] = dict_R[req]
                        # if req is only in rest requests still in assigned pool
                        elif req in rest_requests and req not in done_requests:
                            if req not in assigned_pool:
                                assigned_pool.append(req)
                                dict_R_assigned[req] = dict_R[req]

                    # build the trip from the current location to the destination
                    row_copy = copy.deepcopy(route[last_done_trip])
                    row_copy[0] = depots_loc[k]
                    row_copy[1] = cur_terminal
                    row_copy[2] = float(tt)
                    row_copy[3] = float(tt)
                    row_copy[4] = b_k_temp
                    row_copy[5] = route[last_done_trip+1][4]

                    # remove the trips that are completed and stack the row_copy as the first row
                    route_updated[k] = np.vstack((row_copy, rest_route))

                    # get the sequence of the vessel
                    sequences[k] = seq_from_route(route_updated[k])

                    # print("Complete dwelling update")

                    

        # update the battery level
        b_temp[k] = b_k_temp
    # update the request status
    r_od_ind_assigned = get_o_d_index(dict_R_assigned, terminals)
    r_od_ind_unassigned = get_o_d_index(dict_R_unassigned, terminals)
    r_od_ind_serving = get_o_d_index(dict_R_serving, terminals)

    # create the object for the route sequence
    dict_route = Route(sequences)
    # for later keep it in case the new route is not feasible
    obj_old, dist_seq = dict_route.get_totaldist(sequences, dist_temp)  

    #%% limit the insertion positions of new requests
    start_insertion = time.time()
    # for each unassigned request, check where the last insertion position is for each vessel
    dict_R_unassigned_temp = copy.deepcopy(dict_R_unassigned)
    # position_constraints = {k: {(key, "pickup", "earliest"): True, (key, "pickup", "latest"): True, (key, "delivery", "earliest"): True, (key, "pickup", "latest"): True} for key in dict_R_unassigned_temp.keys() for k in range(n_k)}    # initialise the constraint for each vessel by True, which means no constraint in the insertion position
    position_constraints = {k: {} for k in range(n_k)}

    for key, item in dict_R_unassigned_temp.items():
        pickupend = item['ptw']
        delivery_limit = item['b']
        r_type = item['s']
        service_time = item['st']
        threshold = pickupend - service_time
        
        # check with each vessel
        for k in range(n_k):
            vessel_type = vessels.at[k, 'type']
            if (vessel_type != r_type) and (vessel_type != 2):
                # request cannot be inserted to this vessel's route plan --> False
                position_constraints[k][(key, "pickup", "earliest")] = False
                position_constraints[k][(key, "pickup", "latest")] = False
                position_constraints[k][(key, "delivery", "earliest")] = False
                position_constraints[k][(key, "delivery", "latest")] = False

            # if no route and vessel is available for the request
            elif len(routes[k]) == 0:
                position_constraints[k][(key, "pickup", "earliest")] = True
                position_constraints[k][(key, "pickup", "latest")] = True
                position_constraints[k][(key, "delivery", "earliest")] = True
                position_constraints[k][(key, "delivery", "latest")] = True

            else:
                # get the last node visited before the threshold and the delivery time window
                route = routes[k]
                # pickup and delivery earliest is always True
                position_constraints[k][(key, "pickup", "earliest")] = True
                position_constraints[k][(key, "delivery", "earliest")] = True
                index_last_pickup = False
                try:
                    pickup = [r for r in route if float(r[3]) <= threshold]
                    last_pickup = pickup[-1]
                    if np.where(np.all(route == last_pickup, axis=1))[0][0] == len(pickup) - 1:
                        position_constraints[k][(key, "pickup", "latest")] = True
                    else:
                        index_last_pickup = np.where(np.all(route == last_pickup, axis=1))[0][0] + 2   # the index of the last pickup node in the sequence
                # no room for the pickup
                except IndexError:
                    index_last_pickup = False
                    position_constraints[k][(key, "pickup", "latest")] = False
                    # Therefore all false
                    position_constraints[k][(key, "delivery", "earliest")] = False
                    position_constraints[k][(key, "delivery", "latest")] = False
                    position_constraints[k][(key, "pickup", "earliest")] = False
                if index_last_pickup:
                    delivery = [r for r in route if float(r[3]) <= delivery_limit]
                    last_delivery = delivery[-1]
                    if np.where(np.all(route == last_delivery, axis=1))[0][0] == len(delivery) - 1:
                        position_constraints[k][(key, "delivery", "latest")] = True

                    else:
                        index_last_delivery = np.where(np.all(route == last_delivery, axis=1))[0][0] + 2    # the index of the last delivery node in the sequence
                        position_constraints[k][(key, "delivery", "latest")] = index_last_delivery

    # print("position constraints generated")
#%% get the new best sequence
    # print("Start searching for the best sequence")
    dict_seqdist, dict_tracking = dict_route.get_newsequences(position_constraints, r_od_ind_unassigned, dist_temp)

#%% CHECK check the request status

    # get the best sequence and the route
    best_seq, best_routes = get_best_sequence(tt, dict_seqdist, dict_tracking, dict_R, dict_r_k, dict_R_assigned, dict_R_unassigned, dict_R_serving, r_od_ind_serving, r_od_ind_assigned, r_od_ind_unassigned, B, b_temp, charge_loc, n_k, dist_temp, speed)

    if best_seq is None:
        # print("No feasible route found")
        # print("Keeping the original route plan and new requests are rejected.")
        # keep the original route plan
        sequences = sequences
        routes = route_updated
        obj = obj_old
        
        dict_R_unassigned_copy = copy.deepcopy(dict_R_unassigned)
        for key in dict_R_unassigned_copy.keys():
            key_R_failed.append(key)
            unassigned_pool.remove(key)
            R_unassigned = R_unassigned[R_unassigned['key'] != key]
            del dict_R_unassigned[key]
            del r_od_ind_unassigned[key]
    
    else:
        sequences = best_seq
        routes = best_routes
        
        dict_route_new = Route(sequences)
        obj, dist_seq = dict_route_new.get_totaldist(dict_route_new.sequences, dist_temp)

        
    end_insertion = time.time()
    duration = end_insertion - start_insertion
    
    # print("Insertion done")
    # print(f"Solving time: {round(duration, 4)} seconds")

#%% Update the status of requests and dict_r_k

    # update the assignment of new requests
    if best_seq is not None:
        for tups in dict_tracking[str(best_seq)]:
            dict_r_k[tups[0]] = tups[1]

        # update the request status
        dict_R_unassigned_copy = copy.deepcopy(dict_R_unassigned)
        for key in dict_R_unassigned_copy.keys():
            if key in unassigned_pool:
                unassigned_pool.remove(key)
                assigned_pool.append(key)
                dict_R_assigned[key] = dict_R_unassigned[key]
                del dict_R_unassigned[key]


    return sequences, routes, obj, dict_R, dict_R_assigned, dict_R_unassigned, dict_R_serving, dict_r_k, key_R_served, key_R_failed, depots_temp, depots_temp_all, depots_loc, distances, empty_distances, b_temp, duration, unassigned_pool, assigned_pool, serving_pool


#%% class for route sequence and route construction
class Route():
    def __init__(self, sequences):
        # sequence if a list of terminal indices to be visited in order per vessel
        self.sequences = sequences


    def get_newsequences(self, position_constraints, r_od_ind_unassigned, dist_mat):

        sequences = self.sequences    # an array with 2 lists of seqeunces - old one
        pairs_dict = r_od_ind_unassigned    # a dictionary with the origin and destination indices of the requests
        position_constraints = position_constraints    # a dictionary with the constraints for the insertion positions

        # get the new sequence
        all_possible_lists = self.insert_multiple_pairs_into_multiple_lists(sequences, pairs_dict, position_constraints)


        # get the total distance of the route
        dict_seqdist = {}
        dict_tracking = {}
        for seq, tracking in all_possible_lists:
            total_dist, dist_seq = self.get_totaldist(seq, dist_mat)
            dict_seqdist[str(seq)] = total_dist
            dict_tracking[str(seq)] = tracking    # track in which vessel the new requests were inserted
        

        # sort the dictionary by the total distance
        dict_seqdist = dict(sorted(dict_seqdist.items(), key=lambda item: item[1]))

        return dict_seqdist, dict_tracking


    def get_totaldist(self, sequences, dist_mat):
        """
        Calculate the distance of the route
        """
        # total distance of the route, all vessels
        if type(sequences) == str:
            sequences = ast.literal_eval(sequences)
        
        total_dist = 0
        dist_seq = [0 for _ in range(len(sequences))]
        for j, seq in enumerate(sequences):
            for i in range(len(seq)-1):
                total_dist += dist_mat[seq[i], seq[i+1]]
                dist_seq[j] += dist_mat[seq[i], seq[i+1]]
        
        return total_dist, dist_seq

    def build_route(self, seq, tt, dict_R, r_od_ind_serving, r_od_ind_assigned, r_od_ind_unassigned, dict_r_k, dict_seqdist, dict_tracking, dist_mat, B, b_temp, charge_loc, n_k, speed):
        """
        Function to build the route plan
        """
        
        sequence = ast.literal_eval(seq)    # the sequence of the route
        tracking = dict_tracking[str(seq)]    # the tracking of the route
        for tup in tracking:
            dict_r_k[tup[0]] = tup[1]

        # merge the serving, assigned and unassigned requests
        r_od_ind = {**r_od_ind_serving, **r_od_ind_assigned, **r_od_ind_unassigned}
        # identify in which trips the requests are being served
        boarding = {key: (0, -1) for key in r_od_ind.keys()}    # the index of the node in the sequence where the request board and alight
        # serving requests
        for key in r_od_ind_serving.keys():
            vess = dict_r_k[key]
            for i in range(len(sequence[vess])):
                if sequence[vess][i] == r_od_ind_serving[key][1]:
                    boarding[key] = (0, i)
                    break
        # assigned requests and newly inserted unassigned requests
        for key in (list(r_od_ind_assigned.keys()) + list(r_od_ind_unassigned.keys())):
            vess = dict_r_k[key]
            try:
                ori = r_od_ind_assigned[key][0]
                des = r_od_ind_assigned[key][1]
            except KeyError:
                ori = r_od_ind_unassigned[key][0]
                des = r_od_ind_unassigned[key][1]

            tup_board = [0, -1]
            for i in range(len(sequence[vess])):

                if tup_board[0] == 0:
                    if sequence[vess][i] == ori:
                        tup_board[0] = i
                        break

            for i in range(len(sequence[vess])):
                if tup_board[1] == -1:
                    if sequence[vess][i] == des:
                        if i > tup_board[0]:
                            tup_board[1] = i
                            break

            boarding[key] = tuple(tup_board)
        
        Routes = [0 for _ in range(n_k)]
        # route plan construction
        for k in range(n_k):
            route = np.zeros((len(sequence[k])-1, 8), dtype=object)    # the route plan
            # filter the boarding requests with dict_r_k
            requests = [key for key in boarding.keys() if dict_r_k[key] == k]
            route[:, 2:4] = tt    # the time of the route plan
            route[:, 4:6] = b_temp    # the battery level of the route plan

            for i in range(len(sequence[k])-1):
                # origin and destination of the trip
                serving_requests = [key for key in r_od_ind_serving.keys() if dict_r_k[key] == k]
                route[i, 0] = sequence[k][i]
                route[i, 1] = sequence[k][i+1]

                travel_time = dist_mat[sequence[k][i], sequence[k][i+1]] / speed * 60    # minutes
                service_time = 0
                # add requests to the serving requests
                for req in requests:
                    if i >= boarding[req][0] and i < boarding[req][1]:
                        if req not in serving_requests:
                            serving_requests.append(req)
                            service_time += dict_R[req]['st']

                    elif i >= boarding[req][1]:
                        if req in serving_requests:
                            serving_requests.remove(req)

                # the first trip of the vessel
                if i == 0:
                    dep_time = tt + service_time
                    route[i, 2] = dep_time

                    arrival_time = dep_time + travel_time
                    route[i, 3] = arrival_time
                    route[i, 4] = b_temp[k]
                    route[i, 5] = b_temp[k] - dist_mat[sequence[k][i], sequence[k][i+1]] * 1
                    loading_all = sum([dict_R[key]['qr'] for key in serving_requests])
                    loading_parc = sum([dict_R[key]['qr'] for key in serving_requests if dict_R[key]['s'] == 0])
                    loading_pass = sum([dict_R[key]['qr'] for key in serving_requests if dict_R[key]['s'] == 1])
                    route[i, 6] = [loading_all, loading_pass, loading_parc]
                    route[i, 7] = serving_requests   # the requests being served

                else:
                    # time
                    route[i, 2] = float(route[i-1, 3]) + service_time
                    route[i, 3] = float(route[i, 2]) + travel_time    # departure time + travel time
                    
                    # battery level
                    # origin
                    if sequence[k][i] in charge_loc:
                        route[i, 4] = min(route[i-1, 5] + 100/60 * (route[i, 2] - route[i-1, 3]), B)
                    else:
                        route[i, 4] = route[i-1, 5]

                    # destination
                    route[i, 5] = float(route[i, 4]) - dist_mat[sequence[k][i], sequence[k][i+1]] * 1

                    # loading
                    loading_all = sum([dict_R[key]['qr'] for key in serving_requests])
                    loading_parc = sum([dict_R[key]['qr'] for key in serving_requests if dict_R[key]['s'] == 0])
                    loading_pass = sum([dict_R[key]['qr'] for key in serving_requests if dict_R[key]['s'] == 1])
                    route[i, 6] = [loading_all, loading_pass, loading_parc]
                    route[i, 7] = serving_requests

            Routes[k] = route

        return Routes
                        
    def check_feasibility(self, routes, dict_R, dict_r_k, dict_R_assigned, dict_R_unassigned, dict_R_serving, B, n_k):
        """
        Function to check the feasibility of the route plan
        """

        dict_R_all = {**dict_R_assigned, **dict_R_unassigned, **dict_R_serving}

        dict_R_assigned_unassigned = {**dict_R_assigned, **dict_R_unassigned}

        sequence = self.sequences
        # vessel constraints
        for k in range(n_k):
            route = routes[k]
            for i in range(len(route)):
                # check the battery level
                if route[i, 5] < 0.2 * B:
                    # print("Battery level too low")
                    return False
                if route[i, 4] < 0.2 * B:
                    # print("Battery level too low")
                    return False
                
                # check the loading
                if route[i, 6][0] > 50:
                    # print("Loading capacity exceeded")
                    return False
        # print("Vessel constraints passed")
        # check the time window of assigned and unassigned requests
        for key, item in dict_R_assigned_unassigned.items():
            route = routes[dict_r_k[key]]

            first_trip_ind = 0
            end_trip_ind = 0
            for i in range(len(route)): 
                if key in route[i, 7]:
                    first_trip_ind = i
                    break
            
            for j in range(first_trip_ind, len(route)):
                if key in route[j, 7]:
                    end_trip_ind = j
                
                if key not in route[j, 7]:
                    break
            # check the pickup time window
            if first_trip_ind != 0:
                if route[first_trip_ind-1, 3] > float(item['ptw']):
                    # print(f"Pickup time window violated for request {key}")
                    return False

            # check the delivery time window
            if float(item['b']) < route[end_trip_ind, 3]:
                # print(f"Delivery time window violated for request {key}")
                return False

        # print("Time window constraints passed for assigned and unassigned requests")
        # print([[key, dict_r_k[key]] for key in dict_R_serving.keys()])
        for key, item in dict_R_serving.items():
            route = routes[dict_r_k[key]]
            # print("Vessel", dict_r_k[key], "key", key)
            end_trip_ind = 0
            for j in range(len(route)):
                if key in route[j, 7]:
                    end_trip_ind = j
                
                if key not in route[j, 7]:
                    break
            
            # check the delivery time window
            if float(item['b']) < route[end_trip_ind, 3]:
                # print(f"Delivery time window violated for request {key}")
                return False
        
        # print("Time window constraints passed for serving requests")
        # print("Feasibility check passed")
        return True

    def insert_multiple_pairs_into_multiple_lists(self, original_lists, pairs_dict, position_constraints):
        # Function to insert a single pair while respecting the order and position constraints
        def insert_pair(current_list, pair, pair_key, constraints, list_index):
            results = []
            n = len(current_list)
            first, second = pair

            # Ensure the first element of the original list remains fixed
            fixed_first_element = current_list[0]
            fixed_first_element_list = [fixed_first_element]

            earliest_first_pos = max(1, constraints[list_index].get((pair_key, "pickup", "earliest"), 0))
            max_first_pos = min(n, constraints[list_index].get((pair_key, "pickup", "latest"), n))
            earliest_second_pos = max(1, constraints[list_index].get((pair_key, "delivery", "earliest"), 0))
            max_second_pos = min(n + 1, constraints[list_index].get((pair_key, "delivery", "latest"), n + 1))

            # If the constraint is False, return an empty list
            if earliest_first_pos is False or max_first_pos is False or earliest_second_pos is False or max_second_pos is False:
                return results

            # If the constraint is True, set it to the minimum or maximum possible position
            if earliest_first_pos is True:
                earliest_first_pos = 1
            if max_first_pos is True:
                max_first_pos = n
            if earliest_second_pos is True:
                earliest_second_pos = 1
            if max_second_pos is True:
                max_second_pos = n + 1

            # Iterate over all possible positions for the first element within its constraints
            for i in range(earliest_first_pos, min(max_first_pos + 1, n + 1)):
                # Skip if inserting first element would cause a duplicate
                if current_list[i-1] == first:
                    continue

                # Insert the first element after the fixed first element
                temp_list = fixed_first_element_list + current_list[1:i] + [first] + current_list[i:]

                # Iterate over all possible positions for the second element within its constraints after the first
                for j in range(max(i + 1, earliest_second_pos), min(max_second_pos + 1, n + 2)):
                    # Skip if inserting second element would cause a duplicate
                    if temp_list[j-1] == second:
                        continue

                    # Insert the second element
                    new_list = temp_list[:j] + [second] + temp_list[j:]
                    results.append(new_list)

            return results

        # Recursive function to insert all pairs one by one
        def recursive_insert(current_lists, pairs_keys, pairs_dict, constraints, current_tracking):
            if not pairs_keys:
                return [(current_lists, current_tracking)]  # Base case: return the current lists and tracking as the only result
            
            first_pair_key, rest_pairs_keys = pairs_keys[0], pairs_keys[1:]  # Split the first pair key from the rest
            first_pair = pairs_dict[first_pair_key]
            
            all_results = []
            
            # Try inserting the first pair into each of the current lists
            for idx, current_list in enumerate(current_lists):
                # Insert the pair into the selected list
                intermediate_results = insert_pair(current_list, first_pair, first_pair_key, constraints, idx)
                
                # For each result from the insertion, continue with the rest of the pairs
                for result in intermediate_results:
                    # Copy the current state of lists and update the corresponding list with the result
                    new_current_lists = current_lists[:]
                    new_current_lists[idx] = result
                    new_tracking = current_tracking[:]
                    new_tracking.append((first_pair_key, idx))
                    # Recursively insert the remaining pairs
                    all_results.extend(recursive_insert(new_current_lists, rest_pairs_keys, pairs_dict, constraints, new_tracking))
            
            return all_results

        # Helper function to remove consecutive duplicates from a list
        def remove_consecutive_duplicates(single_list):
            if not single_list:
                return single_list
            result = [single_list[0]]
            for item in single_list[1:]:
                if item != result[-1]:
                    result.append(item)
            return result

        # Generate all possible combinations of inserting each pair into any of the original lists
        pairs_keys = list(pairs_dict.keys())
        all_combinations = recursive_insert(original_lists, pairs_keys, pairs_dict, position_constraints, [])

        # Filter out consecutive duplicates in each list of each combination
        filtered_combinations = []
        for lists, tracking in all_combinations:
            filtered_lists = [remove_consecutive_duplicates(lst) for lst in lists]
            filtered_combinations.append((filtered_lists, tracking))
        
        return filtered_combinations

#%% Out of the class
def seq_from_route(route):
    """
    Function to get the sequence from the route
    """
    seq = []
    seq.append(route[0][0])
    for i in range(len(route)):
        seq.append(route[i][1])
    
    return seq

def get_best_sequence(tt, dict_seqdist, dict_tracking, dict_R, dict_r_k, dict_R_assigned, dict_R_unassigned, dict_R_serving, r_od_ind_serving, r_od_ind_assigned, r_od_ind_unassigned, B, b_temp, charge_loc, n_k, dist_mat, speed):
    """"
    Function to obtain the best feasible route.
    """

    # dict_seqdist is the dictionary with the sequences and the total distance, sorted by the total distance
    for seq in dict_seqdist.keys():
        obj_seq = Route(seq)
        seq = str(obj_seq.sequences)
        # build the route
        route = obj_seq.build_route(seq, tt, dict_R, r_od_ind_serving, r_od_ind_assigned, r_od_ind_unassigned, dict_r_k, dict_seqdist, dict_tracking, dist_mat, B, b_temp, charge_loc, n_k, speed)
        # check the feasibility of the route - if True, that is the next route
        if obj_seq.check_feasibility(route, dict_R, dict_r_k, dict_R_assigned, dict_R_unassigned, dict_R_serving, B, n_k):
            
            seq = ast.literal_eval(seq)
            return seq, route

    # if no feasible route is found
    return None, None



#%% Other functions
"""
Notations
N: terminal locations
K: vehicle set
o: Set of depot locations
R: Set of requests
"""

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

    # x_o, y_o = origin[0], origin[1]
    # x_d, y_d = destination[0], destination[1]

    # # calculate the euclidean distance
    # distance = math.sqrt((x_d - x_o)**2 + (y_d - y_o)**2)
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

def get_temp_distmat(dist_mat, n_k, tt, locs, revisit, terminals):
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

    # get the location of vessels
    # locs = get_location(tt, n_k, terminals_1, routes, depots_temp, depots_loc, distances, empty_distances)[0]   # this is a list of tuples [(k, x, y), ...]
    
    # make a list of coordinates of terminals and vessels
    term_coords = [terminals['N'][i] for i in range(len(terminals))]
    ves_coords = [str(loc[1:]) for loc in locs]

    # print(len(term_coords))
    # print("ves_coords", ves_coords)
    # calculate the distance between all terminals and vessels
    temp_distmat = np.zeros((len(term_coords) + len(ves_coords) + 1, len(term_coords) + len(ves_coords) + 1))
    temp_distmat[:len(term_coords)+1, :len(term_coords)+1] = dist_mat
    
    for i in range(len(term_coords)):
        for j in range(len(ves_coords)):
            # print(i, j+1+len(term_coords))
            dist = get_distance(term_coords[i], ves_coords[j])
            # dist = np.sqrt((term_coords[i][0] - ves_coords[j][0])**2 + (term_coords[i][1] - ves_coords[j][1])**2)
            # print(dist)
            temp_distmat[i, j+len(term_coords)+1] = dist
            temp_distmat[j+len(term_coords)+1, i] = dist
    for i in range(len(ves_coords)):
        for j in range(len(ves_coords)):
            # print(i+len(term_coords)+1, j+len(term_coords)+1)
            # use square distance
            dist = get_distance(ves_coords[i], ves_coords[j])
            # dist = np.sqrt((ves_coords[i][0] - ves_coords[j][0])**2 + (ves_coords[i][1] - ves_coords[j][1])**2)
            # print(dist)
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
    # distance travelled by the vessels
    # distances = [0 for _ in range(n_k)]

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
        # print("vessel", k)
        # print(route)
        try:
            # search for the row that is currently being served
            if tt <= float(route[0][2]):
                # vessel is at the current location
                # print("route not started yet")
                location = int(route[0][0])
                current_location[k] = (k, depots_temp[location][0], depots_temp[location][1])

            elif tt >= float(route[-1][3]):
                # vessel is at the last location
                # print("route completed")
                location = int(route[-1][1])
                current_location[k] = (k, depots_temp[location][0], depots_temp[location][1])

                # loop over all rows in route to calculate the distance travelled
                for r in route:
                    ori = int(r[0])
                    des = int(r[1])
                    distance += get_distance(str(depots_temp[ori]), str(depots_temp[des]))
                    if r[6] == [0, 0, 0]:
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
                    o = depots_temp[ori]
                    d = depots_temp[des]

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
                            if route[ind][6] == [0, 0 ,0]:
                                empty_distance += get_distance(str(depots_temp[ori_ind]), str(depots_temp[des_ind]))

                    # for ind in range(trip_index):
                    #     ori_ind = int(route[ind][0])
                    #     des_ind = int(route[ind][1])
                    #     distance += get_distance(str(depots_temp[ori_ind]), str(depots_temp[des_ind]))
                    
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
                            if r[6] == [0, 0, 0]:
                                empty_distance += get_distance(str(depots_temp[ori_terminal]), str(depots_temp[cur_terminal]))
                        
                        else:
                            break
                    # print("vessel is dwelling at the terminal")
                    # print("distance", distance)
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
    R['st'] = float(0)

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
    # print(keys)
    # print(dict_R)

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
