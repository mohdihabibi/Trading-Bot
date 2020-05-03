import logging

import pandas as pd

from utils.agent import Agent
from utils.methods import train_model, evaluate_model

def format_position(price):
    return ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

def show_train_result(result, val_position, initial_offset):
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
                     .format(result[0], result[1], format_position(result[2]), result[3]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))

def main():
    agent = Agent()

    df = pd.read_csv("data/GOOG.csv")
    train_data = list(df['Adj Close'])
    df = pd.read_csv("data/GOOG_2018.csv")
    val_data = list(df['Adj Close'])

    initial_offset = val_data[1] - val_data[0]
    ep_count = 50
    batch_size = 32
    window_size = 5
    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size)
        show_train_result(train_result, val_result, initial_offset)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted!")
