"""
Passengers requests class
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class Pass_requests():
    def __init__(self, file_path: str):
        
        self.file_path = file_path
        # time length of each interval (minutes) --- sum is 960
        # 6:00-7:00, 7:00-9:30, 9:30-16:00, 16:00-18:30, 18:30-22:00
        self.interval = [60, 150, 390, 150, 210]
        self.total_time = sum(self.interval)
    
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
        # only the ones with "P_"
        self.sheetnames = [sheet for sheet in sheets if "P_" in sheet]


    def all_process(self):
        """
        All the process to generate the dataframe of the requests per timestep
        """
        self.get_parameters()
        self.all_OD_time()
        self.all_request_attributes()
        self.build_df()

        return self.dict_requests_tt

    

    def get_parameters(self):

        # get the parameters from the excel file
        self.parameters = {}
        for sheet in self.sheetnames:
            para = self.data[sheet].set_index(self.data[sheet].columns[0])
            para = para.to_numpy()
            self.parameters[sheet] = para

        return self.parameters

    def single_generate_poisson_events(self, ori: int, des: int, time_index: int):

        sheet = "P_" + str(time_index)
        rate = self.parameters[sheet][ori, des]
        time_duration = self.interval[time_index]

        TT = sum(self.interval[:time_index])
        
        num_events = np.random.poisson(rate * time_duration)
        event_times = np.sort(np.random.uniform(0, time_duration, num_events))
        # round the event times to integer
        event_times = np.round(event_times, 0).astype(int) + TT
        inter_arrival_times = np.diff(event_times)


        # length of the event_times
        N = len(event_times)
        # interger uniform distribution for the number of events
        r_size = np.random.randint(low=1, high=10, size=N, dtype=int)

        self.dict_events[(ori, des, time_index)] = (num_events, event_times, inter_arrival_times, r_size)

        return (num_events, event_times, inter_arrival_times, r_size)

    
    def all_OD_time(self):

        self.dict_events_all = {}

        # for all OD pair
        for t in range(len(self.interval)):
            for i in range(self.n_t):
                for j in range(self.n_t):
                    if i != j:

                        self.dict_events_all[(i, j, t)] = self.single_generate_poisson_events(i, j, t)

        return self.dict_events_all


    def request_attributes(self, ori: int, des: int, time_index: int, ind: int, transit_time=45):
            
            try:
                requests = self.dict_events_all[(ori, des, time_index)]
                r_num = requests[0]
                r_times = requests[1]
                inter_arrival_times = requests[2]
                r_sizes = requests[3]


                # get one of the request
                r = r_times[ind]
                # get the size of the request
                r_size = r_sizes[ind]
                # get the origin and destination of the request
                r_ori = self.terminals["N"][ori]
                r_des = self.terminals["N"][des]
                # get the pickup time of the request
                r_pickup = r_times[ind]
                # get the delivery time of the request
                r_time_del = r_pickup + transit_time
                # type of the request
                r_type = 1
                # get the size of the request
                r_size = r_sizes[ind]
                # get the max transit time of the request
                r_max_transit_time = transit_time

                # create a series of the request
                request_attributes = pd.Series([r_ori, r_des, r_pickup, r_time_del, r_size, r_type, r_max_transit_time], index=['p', 'd', 'a', 'b', 'qr', 's', 'gamma'])
                
                
                return request_attributes
            
            except IndexError:
                print(f"IndexError: OD pair ({ori}, {des}) at time index {time_index} does not have request with index {ind}")

                return None



    def all_request_attributes(self, transit_time=45):
            
            self.dict_events_all = self.all_OD_time()


            self.dict_requests = {}
            self.counter = np.zeros((self.n_t, self.n_t, len(self.interval)))

            for t in range(len(self.interval)):

                for i in range(self.n_t):
                    for j in range(self.n_t):
                        if i != j:
                            counter = 0
                            for ind in range(len(self.dict_events_all[(i, j, t)][1])):
                                self.dict_requests[(i, j, self.dict_events_all[(i, j, t)][1][ind])] = self.request_attributes(i, j, t, ind, transit_time)
                                counter += 1
                            self.counter[i, j, t] = counter
                
    
            return self.dict_requests, self.counter


    def build_df(self):
    
        # get the series of the request with time tt in the third key of the dictionary
        dict_requests_tt = {}

        for tt in range(self.total_time):
            requests_tt = pd.DataFrame(columns=['p', 'd', 'a', 'b', 'qr', 's', 'gamma'])
            for i, key in enumerate(self.dict_requests.keys()):
                if key[2] == tt:

                    requests_tt.loc[i] = self.dict_requests[key]
            if not requests_tt.empty:  
                dict_requests_tt[tt] = requests_tt

        self.dict_requests_tt = dict_requests_tt

        return self.dict_requests_tt


    
    def poisson_simulation(self, rate, time_duration, show_visualization=False):


        if isinstance(rate, float) or isinstance(rate, int):
            num_events, event_times, inter_arrival_times = generate_poisson_events(rate, time_duration)
            
            if show_visualization:
                plot_non_sequential_poisson(num_events, event_times, inter_arrival_times, rate, time_duration)
            else:
                return num_events, event_times, inter_arrival_times

        elif isinstance(rate, list):
            num_events_list = []
            event_times_list = []
            inter_arrival_times_list = []

            for individual_rate in rate:
                num_events, event_times, inter_arrival_times = generate_poisson_events(individual_rate, time_duration)
                num_events_list.append(num_events)
                event_times_list.append(event_times)
                inter_arrival_times_list.append(inter_arrival_times)

            # if show_visualization:
            #     plot_sequential_poisson(num_events_list, event_times_list, inter_arrival_times_list, rate, time_duration)
            # else:
            return num_events_list, event_times_list, inter_arrival_times_list





