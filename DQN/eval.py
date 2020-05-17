"""
Script for evaluating Stock Trading Bot.

Usage:
  eval.py <eval-stock> [--window-size=<window-size>] [--model-name=<model-name>] [--debug]

Options:
  --window-size=<window-size>   Size of the n-day window stock data representation used as the feature vector. [default: 10]
  --model-name=<model-name>     Name of the pretrained model to use (will eval all models in `models/` if unspecified).
  --debug                       Specifies whether to use verbose logs during eval operation.
"""

import os
import coloredlogs
import pandas as pd
import csv
import logging
import plotly.express as px
import plotly.graph_objects as go

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_eval_result,
    switch_k_backend_device
)


def main(eval_stock, window_size, model_name, debug):
    """ Evaluates the stock trading bot.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python eval.py --help]
    """    
    df_google = pd.read_csv("data/GOOG_2019.csv")
    data = list(df_google['Open'])
    data_date = list(df_google['Date'])
    initial_offset = data[1] - data[0]


    agent = Agent(window_size, current_price=data[0],pretrained=True, model_name=model_name)
    total_profit, cash_in_hand, total_share, google_buy, google_sell = evaluate_model(agent, data, data_date, window_size, debug)

    show_eval_result(model_name, total_profit, initial_offset)
        
 

    google_price_buy = []
    google_buy_date = []
    google_price_sell = []
    google_sell_date = []
    

    w = csv.writer(open("dqn.csv", "w"))

    for date, price in google_buy:
        google_price_buy.append(price)
        google_buy_date.append(date)
        w.writerow(['Buy', date, price]) 

    for date, price in google_sell:
        google_price_sell.append(price)
        google_sell_date.append(date)
        w.writerow(['Sell', date, price])

    fig = px.line(df_google, x='Date', y='Open')
    fig.add_trace(go.Scatter(x=google_buy_date, y=google_price_buy, mode="markers", showlegend=True, name="Buy"))
    fig.add_trace(go.Scatter(x=google_sell_date, y=google_price_sell, mode="markers", showlegend=True, name="Sell"))
    fig.update_layout(title="DQN - Test results on Goog_2019 stock data with profit of " + str(total_profit),font=dict(
        size=9,
        color="#7f7f7f"
    ))
    fig.show()            


if __name__ == "__main__":
    args = docopt(__doc__)

    eval_stock = args["<eval-stock>"]
    window_size = int(args["--window-size"])
    model_name = args["--model-name"]
    debug = args["--debug"]

    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()

    try:
        main(eval_stock, window_size, model_name, debug)
    except KeyboardInterrupt:
        print("Aborted")
