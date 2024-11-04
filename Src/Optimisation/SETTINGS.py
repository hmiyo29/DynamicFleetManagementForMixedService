"""
Setting the configuration for the simulation

"""
import os
import re


# data folder
data_folder = "C:/Users/heisu/Desktop/Thesis/MYCODE/data_pilot_3/"

# set the output folder
output_folder = "C:/Users/heisu/Desktop/Thesis/MYCODE/output_insertion/"

class Param:
    def __init__(self, data_folder, output_folder):
        self.data_folder = data_folder
        self.output_folder = output_folder

    def set_param(self, i:int):
        self.scenario = self.configurations[i][0]
        self.inst = self.configurations[i][1]
        self.pattern = self.configurations[i][2]

    def configs(self):
    # N: total number of simulations
    
    
    # vessel service possibilities
    # patterns = [(2, 2), (2, 1), (1, 2), (0, 2), (2, 0), (1, 0), (0, 1)]

        # find the files in the folder with name starting with 'Requests'
        files = os.listdir(self.data_folder)
        files = [f for f in files if re.match(r'Requests.*\.xlsx', f)]
        files = sorted(files)

        self.files = files

        # create a list of configurations
        configurations = []
        for file in files:
            configuration = file.split('.')[0].split('_')[1:]
            scenario = configuration[5]
            inst = int(configuration[7])
            pattern = [int(configuration[1]), int(configuration[3])]
            configurations.append((scenario, inst, pattern))

        self.configurations = configurations
        self.N = len(configurations)
    
