"""
Script for training Stock Trading Bot.

Usage:
  train.py <train-stock> [--strategy=<strategy>]
    [--window-size=<window-size>] [--batch-size=<batch-size>]
    [--episode-count=<episode-count>] [--model-name=<model-name>]
    [--pretrained] [--debug]

Options:
  --window-size=<window-size>       Size of the n-day window stock data representation
                                    used as the feature vector. [default: 10]
  --batch-size=<batch-size>         Number of samples to train on in one mini-batch
                                    during training. [default: 32]
  --episode-count=<episode-count>   Number of trading episodes to use for training. [default: 50]
  --model-name=<model-name>         Name of the pretrained model to use. [default: model_debug]
  --pretrained                      Specifies whether to continue training a previously
                                    trained model (reads `model-name`).
  --debug                           Specifies whether to use verbose logs during eval operation.
"""

import logging
import coloredlogs

from docopt import docopt

from trading_bot.agent import Agent
from trading_bot.methods import train_model, evaluate_model
from trading_bot.utils import (
    get_stock_data,
    format_currency,
    format_position,
    show_train_result,
    switch_k_backend_device
)


def main(train_stock, window_size, batch_size, ep_count,
         strategy="dqn", model_name="dqn", pretrained=False,
         debug=False):
    """ Trains the stock trading bot using Deep Q-Learning.
    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: [python train.py --help]
    """
    train_data = get_stock_data(train_stock)
    #val_data = get_stock_data(val_stock)
    agent = Agent(window_size, current_price=train_data[0],strategy=strategy, pretrained=pretrained, model_name=model_name)

    #initial_offset = val_data[1] - val_data[0]

    cash_in_hand = []
    total_profit = []

    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        #val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result)
       
        total_profit.append(train_result[2])
        cash_in_hand.append(agent.cash_in_hand)
        agent.reset()    
    
    plt.plot(np.array(cash_in_hand))
    plt.xlabel('cash in hand for each episode')
    plt.ylabel('Amount')
    plt.show()
    plt.plot(np.array(total_profit))
    plt.xlabel('total profit for each episode')
    plt.ylabel('Amount')
    plt.show()

if __name__ == "__main__":

    args = docopt(__doc__)

    train_stock = args["<train-stock>"]
    #val_stock = args["<val-stock>"]
    strategy = args["--strategy"]
    window_size = int(args["--window-size"])
    batch_size = int(args["--batch-size"])
    ep_count = int(args["--episode-count"])
    model_name = args["--model-name"]
    pretrained = args["--pretrained"]
    debug = args["--debug"]


    coloredlogs.install(level="DEBUG")
    switch_k_backend_device()
    
    try:
        main(train_stock, window_size, batch_size,
             ep_count, strategy=strategy, model_name=model_name, 
             pretrained=pretrained, debug=debug)
    except KeyboardInterrupt:
        print("Aborted!")
