import logging
import matplotlib.pyplot as plt
import pandas as pd
from tensorboardX import SummaryWriter
from utils.agent import Agent
from utils.methods import train_model, evaluate_model
from utils.model import save
import numpy as np


def format_position(price):
    return ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

# def show_train_result(result, val_position, initial_offset):
#     if val_position == initial_offset or val_position == 0.0:
#         logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}'
#                      .format(result[0], result[1], format_position(result[2]), result[3]))
#     else:
#         logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f})'
#                      .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3],))


def main():
    writer = SummaryWriter()
    df = pd.read_csv("data/GOOG.csv")
    train_data = list(df['Open'])
    agent = Agent(current_price=train_data[0])
    ep_count = 1000
    batch_size = 32
    window_size = 5
    cash_in_hand = []
    total_profit = []
    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        if episode % 100 == 0:
            save(agent.model, episode)
        total_profit.append(train_result[2])
        cash_in_hand.append(agent.cash_in_hand)
        writer.add_scalar(
            "average loss per episode", train_result[3], episode)
        writer.add_scalars(
            "total profit, cash in hand, number of stocks at the end of each episode",
            {"total profit": train_result[2],
                "cash in hand": agent.cash_in_hand},
            episode)
        writer.add_histogram("number of stocks", agent.total_share, episode)
        agent.reset()
    plt.plot(np.array(cash_in_hand))
    plt.xlabel('cash in hand for each episode')
    plt.ylabel('Amount')
    plt.show()
    plt.plot(np.array(total_profit))
    plt.xlabel('total profit for each episode')
    plt.ylabel('Amount')
    plt.show()
    writer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted!")
