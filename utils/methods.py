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


def get_state(data, apple_data, agent, t, window_size):
    if t < len(data)-1 and t < len(apple_data):
        block = data[t - window_size: t]
        block_mean = np.array(block).mean()
        block_apple = apple_data[t - window_size: t]
        block_apple_mean = np.array(block_apple).mean()
        res = [agent.total_share_goog, agent.total_share_apple, agent.cash_in_hand, data[t], apple_data[t],
               agent.total_share_goog*data[t]+agent.total_share_apple*apple_data[t], block_mean, block_apple_mean]
        for i in range(len(res)):
            res[i] = sigmoid(res[i])
        return np.array([res])
    else:
        raise ValueError


def train_model(agent, episode, data, apple_data, ep_count=100, batch_size=32, window_size=5):
    total_profit = 0
    data_length = len(data) - 1

    agent.reset()
    avg_loss = []

    # state = get_state(data, 0, window_size + 1)
    state = get_state(data, apple_data, agent, 5, window_size)
    # print("train_model___initial state: {}".format(state))
    for t in tqdm(range(0, data_length, window_size), total=data_length/window_size, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
        try:
            reward = 0
            next_state = get_state(
                data, apple_data, agent, t+window_size, window_size)
            action = agent.act(state)
            # BUY
            if action == 1:
                if agent.cash_in_hand < (data[t] + apple_data[t]):
                    print('bankrupt...')
                    agent.remember(state, action, 0, next_state, True)
                    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))
                if len(agent.inventory_goog) > 0:
                    reward = (
                        reward+1) if data[t] < agent.inventory_goog[-1] else 0
                if len(agent.inventory_apple) > 0:
                    reward = (
                        reward+1) if apple_data[t] < agent.inventory_apple[-1] else 0
                agent.cash_in_hand -= data[t]
                agent.cash_in_hand -= apple_data[t]
                agent.total_share_apple += 1
                agent.total_share_goog += 1
                agent.inventory_goog.append(data[t])
                agent.inventory_apple.append(apple_data[t])

            # SELL
            elif action == 2:
                if agent.total_share_goog > 0:
                    if len(agent.inventory_goog) == 0:
                        bought_price = data[0]
                    else:
                        bought_price = agent.inventory_goog.pop(0)
                    delta = data[t] - bought_price
                    reward = reward+1 if delta > 0 else 0
                    agent.cash_in_hand = agent.cash_in_hand + bought_price + delta
                    agent.total_share_goog -= 1
                    total_profit += delta
                if agent.total_share_apple > 0:
                    if len(agent.inventory_apple) == 0:
                        bought_price = apple_data[0]
                    else:
                        bought_price = agent.inventory_apple.pop(0)
                    delta = apple_data[t] - bought_price
                    reward = reward+1 if delta > 0 else 0
                    agent.cash_in_hand = agent.cash_in_hand + bought_price + delta
                    agent.total_share_apple -= 1
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


def evaluate_model(agent, data, apple_data, window_size, debug=True):
    total_profit = 0
    data_length = len(data) - 1

    history = []
    agent.reset()

    state = get_state(data, apple_data, agent, 5, window_size)
    for t in range(0, data_length, window_size):
        try:
            next_state = get_state(
                data, apple_data, agent, t+window_size, window_size)
        except ValueError:
            return total_profit, agent.cash_in_hand, agent.total_share_goog, agent.total_share_apple

        # select an action
        # print("evaluate_model___state: {}".format(state))
        action = agent.act(state, is_eval=True)
        # BUY
        if action == 1:
            if agent.cash_in_hand < (data[t] + apple_data[t]):
                print('bankrupt...')
                return (total_profit)
            agent.cash_in_hand -= data[t]
            agent.cash_in_hand -= apple_data[t]
            agent.total_share_apple += 1
            agent.total_share_goog += 1
            agent.inventory_goog.append(data[t])
            agent.inventory_apple.append(apple_data[t])
            history.append((data[t], "BUY"))
            history.append((apple_data[t], "BUY"))
            print("Buy Google at: {}".format(format_currency(data[t])))
            print("Buy Apple at: {}".format(format_currency(apple_data[t])))

        # SELL
        elif action == 2 and agent.total_share > 0:
            if agent.total_share_goog > 0:
                if len(agent.inventory_goog) == 0:
                    bought_price = data[0]
                else:
                    bought_price = agent.inventory_goog.pop(0)
                delta = data[t] - bought_price
                agent.cash_in_hand = agent.cash_in_hand + bought_price + delta
                agent.total_share_goog -= 1
                total_profit += delta
                history.append((data[t], "SELL"))
                print("Sell Google at: {} | Position: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
            if agent.total_share_apple > 0:
                if len(agent.inventory_apple) == 0:
                    bought_price = apple_data[0]
                else:
                    bought_price = agent.inventory_apple.pop(0)
                delta = apple_data[t] - bought_price
                agent.cash_in_hand = agent.cash_in_hand + bought_price + delta
                agent.total_share_apple -= 1
                total_profit += delta
                history.append((apple_data[t], "SELL"))
                print("Sell Apple at: {} | Position: {}".format(
                    format_currency(apple_data[t]), format_position(apple_data[t] - bought_price)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))

        state = next_state
        print("cash_in_hand: {}, total_profit: {}, total_share Google: {}, total_share Apple: {}".format(
            agent.cash_in_hand, total_profit, agent.total_share_goog, agent.total_share_apple))
    return total_profit, agent.cash_in_hand, agent.total_share_goog, agent.total_share_apple
