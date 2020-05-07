
from utils.agent import Agent
from utils.methods import evaluate_model
import pandas as pd
import logging


def format_position(price):
    return ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


def main():
    df = pd.read_csv("data/GOOG_2019.csv")
    data = list(df['Open'])
    df = pd.read_csv("data/AAPL_2019.csv")
    apple_data = list(df['Open'])
    initial_offset = data[1] - data[0]
    window_size = 5
    agent = Agent(
        current_price_goog=data[0], current_price_apple=apple_data[0], pretrained=True)
    print("Portfolio value before trading bot taking over: {}".format(agent.cash_in_hand +
                                                                      agent.total_share_apple*apple_data[0]+agent.total_share_goog*data[0]))
    total_profit, cash_in_hand, total_share_goog, total_share_apple = evaluate_model(
        agent, data, apple_data, window_size)
    print("Total Profit: {} Cash in hand: {} total share Google: {} total share Apple: {}".format(
        total_profit, cash_in_hand, total_share_goog, total_share_apple))
    print("Portfolio value after trading bot took over: {}".format(cash_in_hand +
                                                                   total_share_apple*apple_data[-1]+total_share_goog*data[-1]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted")
