import os
import logging

import numpy as np

from tqdm import tqdm
import math
from utils.model import save

# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

def sigmoid(x):
    """Performs sigmoid operation
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)

# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))

def get_state(data, agent, t, window_size):
    if t < len(data)-1:
        block = data[t - window_size: t]
        block_mean = np.array(block).mean()
        res = [agent.total_share, agent.cash_in_hand, data[t], agent.total_share*data[t], block_mean]
        return np.array([res])
    else:
        raise ValueError


def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=5):
    total_profit = 0
    data_length = len(data) - 1

    agent.reset()
    avg_loss = []

    # state = get_state(data, 0, window_size + 1)
    state = get_state(data, agent, 5, window_size)
    # print("train_model___initial state: {}".format(state))
    for t in tqdm(range(0, data_length, window_size), total=data_length/window_size, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        try:
            reward = 0
            next_state = get_state(data, agent, t+window_size, window_size)
            # print("train_model___initial state: {}".format(state))
            # select an action
            action = agent.act(state)
            # BUY
            if action == 1:
                if agent.cash_in_hand < data[t]:
                    agent.remember(state, action, float('-inf'), next_state, True)
                    raise ValueError
                agent.cash_in_hand -= data[t]
                agent.total_share += 1
                agent.inventory.append(data[t])

            # SELL
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                delta = data[t] - bought_price
                reward = delta #max(delta, 0)
                agent.cash_in_hand = agent.cash_in_hand + bought_price + delta
                agent.total_share -= 1
                total_profit += delta

            # HOLD
            else:
                pass

            agent.remember(state, action, reward, next_state, False)

            if len(agent.memory) > batch_size:
                loss = agent.train_experience_replay(batch_size)
                avg_loss.append(loss)

            state = next_state
        except IndexError:
            print('here')
            agent.remember(state, action, reward, next_state, True)
            break

    if episode % 10 == 0:
        save(agent.model, episode)

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug=True):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.reset()
    
    state = get_state(data, agent, 5, window_size)

    for t in range(0, data_length, window_size):
        reward = 0
        next_state = get_state(data, agent, t+window_size, window_size)
        
        # select an action
        # print("evaluate_model___state: {}".format(state))
        action = agent.act(state, is_eval=True)
        # BUY
        if action == 1:
            if agent.cash_in_hand < data[t]:
                raise ValueError
            agent.cash_in_hand -= data[t]
            agent.total_share += 1
            agent.inventory.append(data[t])

            history.append((data[t], "BUY"))
            if debug:
                logging.debug("Buy at: {}".format(format_currency(data[t])))
        
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            delta = data[t] - bought_price
            reward = delta #max(delta, 0)
            agent.cash_in_hand = agent.cash_in_hand + bought_price + delta
            agent.total_share -= 1
            total_profit += delta

            history.append((data[t], "SELL"))
            if debug:
                logging.debug("Sell at: {} | Position: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))

        done = (t == data_length)
        agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:
            return total_profit, history
