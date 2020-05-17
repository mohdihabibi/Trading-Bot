import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import (
    format_currency,
    format_position
)
from .ops import (
    get_state
)


def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=5):
    total_profit = 0
    data_length = len(data) - 1

    agent.reset()
    avg_loss = []
    #print('window_size in train_model',window_size)

    #state = get_state(data, 0, window_size + 1)
    state = get_state(data, agent, 5, window_size)
    #print('current state',state)

    for t in tqdm(range(0, data_length, window_size), total=data_length/window_size, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):   
        try:     
            reward = 0
            #next_state = get_state(data, t + 1, window_size + 1)
            next_state = get_state(data, agent, t+window_size, window_size)
            #print('next state',next_state)

            # select an action
            action = agent.act(state)
            #print(action)
            done = (t == data_length - 1)
            #print('before action',agent.cash_in_hand)
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
                if delta > 0:
                   reward = 1
                else:   
                   reward = 0 
                agent.cash_in_hand = agent.cash_in_hand + bought_price + delta
                agent.total_share -= 1
                total_profit += delta

            # HOLD
            else:
                pass
            #print('after action',agent.cash_in_hand)   

            #done = (t == data_length - 1)
            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) > batch_size:
                loss = agent.train_experience_replay(batch_size)
                avg_loss.append(loss)

            state = next_state
        except (IndexError, ValueError):
                agent.remember(state, action, reward, next_state, True)
                return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))
       

    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


def evaluate_model(agent, data, data_date, window_size, debug):
    total_profit = 0
    data_length = len(data) - 1

    google_buy = []
    google_sell = []

    history = []
    agent.reset()
    
    #state = get_state(data, 0, window_size + 1)
    state = get_state(data, agent, 5, window_size)

    for t in range(0, data_length, window_size):       
        #next_state = get_state(data, t + 1, window_size + 1)
        try:
            next_state = get_state(data, agent, t+window_size, window_size)
        except ValueError:
            return total_profit, agent.cash_in_hand, agent.total_share, google_buy, google_sell   
        
        # select an action
        action = agent.act(state, is_eval=True)

        # BUY
        if action == 1:
            if agent.cash_in_hand < data[t]:
                return total_profit, agent.cash_in_hand, agent.total_share, google_buy, google_sell
            google_buy.append((data_date[t], data[t]))    
            agent.cash_in_hand -= data[t]
            agent.total_share += 1
            agent.inventory.append(data[t])

            history.append((data[t], "BUY"))
            if debug:
                print("Buy at: {}".format(format_currency(data[t])))
        
        # SELL
        elif action == 2 and agent.total_share > 0:
            if len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
            else:
                bought_price = data[0]    
            google_sell.append((data_date[t], data[t]))    
            delta = data[t] - bought_price
            reward = delta #max(delta, 0)
            agent.cash_in_hand = agent.cash_in_hand + bought_price + delta
            agent.total_share -= 1
            total_profit += delta

            history.append((data[t], "SELL"))
            if debug:
                print("Sell at: {} | Position: {}".format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        # HOLD
        else:
            history.append((data[t], "HOLD"))

        done = (t == data_length - 1)

        state = next_state
        print("cash_in_hand: {}, total_profit: {}, total_share: {}".format(agent.cash_in_hand, total_profit, agent.total_share))
        if done:
            return total_profit, agent.cash_in_hand, agent.total_share , google_buy, google_sell
