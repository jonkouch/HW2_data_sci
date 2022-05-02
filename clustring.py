import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(2)





def sum_scaling(values):
    min_val = min(values)
    sum = sum(values)
    for j in range(len(values)):
        values[j] = (values[j]-min_val)/sum


def transform_data(data_frame, features):
    data_frame[features].apply(sum_scaling)
    return data_frame[features].to_numpy()
