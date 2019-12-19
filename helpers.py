import numpy as np
import math

def generate_counts_dict(data_list):
    data_dict = {}
    for element in data_list:
        if element in data_dict:
            data_dict[element] = data_dict[element] + 1
        else:
            data_dict[element] = 1
    return data_dict
