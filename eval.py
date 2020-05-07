
from utils.agent import Agent
from utils.methods import evaluate_model
import pandas as pd
import logging

def format_position(price):
    return ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

def show_eval_result(model_name, profit, initial_offset):
    if profit == initial_offset or profit == 0.0:
        print('{}: USELESS\n'.format(model_name))
    else:
        print('{}: {}\n'.format(model_name, format_position(profit)))

def main():
    df = pd.read_csv("data/GOOG_2019.csv")
    data = list(df['Open'])
    initial_offset = data[1] - data[0]
    window_size = 5
    agent = Agent(pretrained=True, current_price=data[0])
    profit, _ = evaluate_model(agent, data, window_size)
    show_eval_result("DDQN_1000", profit, initial_offset)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted")