import os
import logging

import numpy as np

from tqdm import tqdm
import math
from utils.model import save

# Formats Position


def format_position(price): return (
    '-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


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
def format_currency(price): return '${0:.2f}'.format(abs(price))


def get_state(data, agent, t, window_size):
    if t < len(data)-1:
        block = data[t - window_size: t]
        block_mean = np.array(block).mean()
        res = [agent.total_share, agent.cash_in_hand,
               data[t], agent.total_share*data[t], block_mean]
        for i in range(len(res)):
            res[i] = sigmoid(res[i])
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
                    print('bankrupt...')
                    agent.remember(state, action, 0, next_state, True)
                    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))
                if len(agent.inventory) > 0:
                    reward = 1 if data[t] < agent.inventory[-1] else 0
                agent.cash_in_hand -= data[t]
                agent.total_share += 1
                agent.inventory.append(data[t])

            # SELL
            elif action == 2 and agent.total_share > 0:
                if len(agent.inventory) == 0:
                    bought_price = data[0]
                else:
                    bought_price = agent.inventory.pop(0)
                delta = data[t] - bought_price
                reward = 1 if delta > 0 else 0
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
        except (IndexError, ValueError):
            agent.remember(state, action, reward, next_state, True)
            return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, window_size, debug=True):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.reset()

    state = get_state(data, agent, 5, window_size)
    for t in range(0, data_length, window_size):
        try:
            next_state = get_state(data, agent, t+window_size, window_size)
        except ValueError:
            return total_profit, agent.cash_in_hand, agent.total_share

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
            print("Buy at: {}".format(format_currency(data[t])))

        # SELL
        elif action == 2 and agent.total_share > 0:
            if len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
            else:
                bought_price = data[0]
            delta = data[t] - bought_price
            agent.cash_in_hand = agent.cash_in_hand + bought_price + delta
            agent.total_share -= 1
            total_profit += delta

            history.append((data[t], "SELL"))
            print("Sell at: {} | Position: {}".format(
                format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))

        state = next_state
        print("cash_in_hand: {}, total_profit: {}, total_share: {}".format(
            agent.cash_in_hand, total_profit, agent.total_share))
    return total_profit, agent.cash_in_hand, agent.total_share
