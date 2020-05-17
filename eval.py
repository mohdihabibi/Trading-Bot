
from utils.agent import Agent
from utils.methods import evaluate_model
import pandas as pd
import logging
import plotly.express as px
import plotly.graph_objects as go

def format_position(price):
    return ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))


def main():
    df_google = pd.read_csv("data/GOOG_2019.csv")
    data = list(df_google['Open'])
    data_date = list(df_google['Date'])
    window_size = 5
    agent = Agent(
        current_price=data[0], pretrained=True)
    print("Portfolio value before trading bot taking over: {}".format(agent.cash_in_hand +
                                                                      agent.total_share*data[0]))
    total_profit, cash_in_hand, total_share, google_buy, google_sell = evaluate_model(
        agent, data, data_date, window_size)

    google_price_buy = []
    google_buy_date = []
    google_price_sell = []
    google_sell_date = []

    for date, price in google_buy:
        google_price_buy.append(price)
        google_buy_date.append(date)

    for date, price in google_sell:
        google_price_sell.append(price)
        google_sell_date.append(date)

    fig = px.line(df_google, x='Date', y='Open')
    fig.add_trace(go.Scatter(x=google_buy_date, y=google_price_buy, mode="markers", showlegend=True, name="Buy"))
    fig.add_trace(go.Scatter(x=google_sell_date, y=google_price_sell, mode="markers", showlegend=True, name="Sell"))
    fig.update_layout(title="DDQN - single stock - Test results on Goog_2019 stock data with profit of " + str(total_profit),font=dict(
        size=9,
        color="#7f7f7f"
    ))
    fig.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Aborted")
