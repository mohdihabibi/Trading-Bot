import os
import math
import logging

import numpy as np


def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


"""def get_state(data, t, n_days):
    #Returns an n-day state representation ending at time t
    
    d = t - n_days + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
    res = []
    for i in range(n_days - 1):
        #res.append(sigmoid(block[i + 1] - block[i]))
        res.append(block[i + 1] - block[i])
    return np.array([res])"""
def get_state(data, agent, t, window_size):
    #print('len(data)',len(data))
    #print('t',t)
    #print('window_size',window_size)
    if t < len(data)-1:
        block = data[t - window_size: t]
        block_mean = np.array(block).mean()
        res = [sigmoid(agent.total_share), sigmoid(agent.cash_in_hand), sigmoid(data[t]), sigmoid(agent.total_share*data[t]), sigmoid(block_mean)]
        #print('res',res)
        return np.array([res])
    else:
        print('error in get_state')
        raise ValueError    
