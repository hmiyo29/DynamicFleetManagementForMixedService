"""
Parcel requests class
"""

import numpy as np
import matplotlib.pyplot as plt    # if you want to plot the data
import pandas as pd

class Par_requests():
    """
    Class for generating parcel requests
    """
    def __init__(self, file_path):
        
        self.file_path = file_path
        # time length of each interval (minutes) --- sum is 960

        # request arrival time intervals
        # 6:00-12:00, 12:00-16:00, 16:00-22:00 (360, 600, 960)

        # request popup time intervals
        # 6:00-11:00, 11:00-15:00, 15:00-21:00 (300, 540, 900)  
        self.interval = [300, 240, 360]
        self.total_time = sum(self.interval)
        self.arrival_time = [360, 600, 960]

        # probability of request arrival time in each interval
        self.prob_0 = [0.6, 0.2, 0.2]
        self.prob_1 = [0, 0.6, 0.4]
        self.prob_2 = [0, 0, 1]
        self.prob_array = [self.prob_0, self.prob_1, self.prob_2]
    
        # read the whole excel file
        self.data = pd.read_excel(self.file_path, sheet_name=None)
        
        # divide it into different dataframes per sheet
        self.terminals = (self.data['N_fred'])
        self.vessels = (self.data['K'])
        self.depots = (self.data['o'])

        self.n_t = len(self.terminals)
        self.n_k = len(self.vessels)

        self.dict_events = {}

        # get the sheetnames of the excel file
        sheets = list(self.data.keys())
        # only the ones with "F_" --- This is the total number of parcel requests between two terminals in the time period
        self.sheetnames = [sheet for sheet in sheets if "F_" in sheet]

    def generate_requests(self):

        # divide the amount of parcel requests into individual requests' popup time
        # for each terminal pair
        for t in range(len(self.interval)):
            sheet = "F_" + str(t)
            # get the amount of parcel requests between two terminals in the time period
            amount = self.data[sheet].set_index(self.data[sheet].columns[0])
            amount = amount.to_numpy()
            for i in range(self.n_t):
                for j in range(self.n_t):
                    if i != j:
                        num_requests = amount[i, j]
                        time_duration = self.interval[t]

                        TT = sum(self.interval[:t])

                        # generate the popup time of the parcel requests
                        popup_time = np.sort(np.random.uniform(0, time_duration, num_requests))
                        # round the event times to integer
                        popup_time = np.round(popup_time, 0).astype(int) + TT
                        # the size of the parcel requests is a uniform distribution between 1 to 3
                        r_size = np.random.randint(low=1, high=10, size=num_requests, dtype=int)

                        self.dict_events[(i, j, t)] = (num_requests, popup_time, r_size)

        return self.dict_events


    def request_attributes(self, ori, des, time_index, ind):
            
        try:
            requests = self.dict_events[(ori, des, time_index)]
            r_num = requests[0]
            r_times = requests[1]
            r_sizes = requests[2]

            prob = self.prob_array[time_index]


            # get one of the request
            r = r_times[ind]
            # get the size of the request
            r_size = r_sizes[ind]
            # get the origin and destination of the request
            r_ori = self.terminals["N"][ori]
            r_des = self.terminals["N"][des]
            # get the pickup time of the request
            r_pickup = r_times[ind]
            # get the delivery time of the request --- expected arrival time
            arrival_time = np.random.choice(self.arrival_time, p=prob)
            r_time_del = arrival_time
            # type of the request --- 0 for parcel
            r_type = int(0)
            # get the max transit time of the request
            r_max_transit_time = self.total_time

            # create a series of the request
            request_attributes = pd.Series([r_ori, r_des, r_pickup, r_time_del, r_size, r_type, r_max_transit_time], index=['p', 'd', 'a', 'b', 'qr', 's', 'gamma'])
            
            return request_attributes
        
        except IndexError:
            print(f"IndexError: OD pair ({ori}, {des}) at time index {time_index} does not have request with index {ind}")

            return None


    def all_request_attributes(self):
            
        self.dict_requests = {}
        self.counter = np.zeros((self.n_t, self.n_t, len(self.interval))) 
        for t in range(len(self.interval)):
            for i in range(self.n_t):
                for j in range(self.n_t):
                    if i != j:
                        counter = 0
                        for ind in range(len(self.dict_events[(i, j, t)][1])):
                            self.dict_requests[(i, j, self.dict_events[(i, j, t)][1][ind])] = self.request_attributes(i, j, t, ind)
                            counter += 1
                        self.counter[i, j, t] = counter

        return self.dict_requests, self.counter

    def build_df(self):
    
        # get the series of the request with time tt in the third key of the dictionary
        dict_requests_tt = {}
        tt_list = np.array([key[2] for key in self.dict_requests.keys()])
        tt_unique = np.unique(tt_list)

        for tt in tt_unique:
            requests_tt = pd.DataFrame(columns=['p', 'd', 'a', 'b', 'qr', 's', 'gamma'])
            for i, key in enumerate(self.dict_requests.keys()):
                if key[2] == tt:
                    requests_tt.loc[i] = self.dict_requests[key]

            dict_requests_tt[tt] = requests_tt


        self.dict_requests_tt = dict_requests_tt

        return self.dict_requests_tt

    def all_process(self):
        """
        All the process to generate the dataframe of the requests per timestep
        """
        self.generate_requests()
        self.all_request_attributes()
        self.build_df()

        return self.dict_requests_tt
