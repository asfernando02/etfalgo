
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
from matplotlib import pyplot as plt
import pandas_ta
import pandas_ta as ta
from dateutil.relativedelta import relativedelta
import requests
url = 'https://docs.googlefcom/spreadsheets/d/11rkSB-xpJwbfiwZw5Lsbo11xvYg2qL37/export?format=csv'

df = pd.read_csv(url).sort_values('Symbol')

etf_tickers = df.iloc[:,0].tolist()
etf_names = df.iloc[:,1].tolist()
print(etf_tickers)
print(etf_names)
len(etf_tickers)

#Get a list of the tickers from the table above, add any missing tickers
etf_list_2024 = df['Symbol'].tolist()
etf_df = pd.DataFrame(etf_list_2024)
etf_tickers = etf_list_2024.extend(['SOXL', 'TQQQ', 'SCHD', 'QYLD'])

#use the yfinance function .Ticker in its own function to get daily data over the course of a year
def get_stock_data(etf,start,end,interval):
  etf_data = yf.Ticker(etf)
  df = etf_data.history(start= today - timedelta(weeks=52), end= today, interval= interval)
  return df

#use the yfinance function .Ticker in its own function to get weekly data over the course of a year
def get_tenyear_stock_data(etf,start,end,interval):
  etf_data = yf.Ticker(etf)
  df = etf_data.history(start= today - relativedelta(years=10), end= today, interval= interval)
  return df

import random

def graph(df,stock):
    fig = plt.figure(figsize = [15,6])
    ax = plt.subplot(1,1,1)

    ax.plot(df.index, df['Close'], color = 'black', label = 'Close')


    columns_for_ma = df.columns[7:]
    for column in columns_for_ma:
        ax.plot(df.index, df[column], label=column)

    ax.legend(loc = 'upper right')
    ax.set_xlabel('Date')
    ax.set_title(stock)
    plt.show()

def add_MA(df, days):
  column_name = 'MA' + str(days)
  df[column_name] = df['Close'].rolling(int(days)).mean()
  return df

def add_average_volume(df, days):
  column_name = 'volume average' + str(days)
  df[column_name] = df['Volume'].rolling(int(days)).mean()
  return df

def add_lineK(df):
  df['K'] = (df['Close'] - df['Low'].rolling(14).min())/(df['High'].rolling(14).max()-df['Low'].rolling(14).min())*100
  return df

def add_lineD(df):
  df['D'] = df['K'].rolling(int(3)).mean()
  return df

#returns profit for Column Q: 5MA strategy
def wealth_return(df):
    #df = df.dropna()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    #Add a new column "Shares", if MA10>MA50, denote as 1 (long one share of stock), otherwise, denote as 0 (do nothing)
    #df['Shares'] = [1 if df.loc[ei, 'MA10'] > df.loc[ei, 'MA50'] else (-1 if df.loc[ei, 'MA10'] < df.loc[ei, 'MA50'] else 0) for ei in df.index]
    df['Shares'] = [1 if df.loc[ei, 'MA10']>df.loc[ei, 'MA50'] else 0 for ei in df.index]
    # restructuring such that if share == 0 then only proceed !!
    if df.iloc[0,-1] == 1:
        for i in range(len(df)):
            if df.iloc[i, -1] == 0:
                df = df.iloc[i:, :]
                break

    df['Close1'] = df['Close'].shift(-1)
    df['Profit'] = [(df.loc[ei, 'Close1'] - df.loc[ei, 'Close'])
                       if df.loc[ei, 'Shares']==1 else 0 for ei in df.index]
    df['wealth'] = df['Profit'].cumsum()
    #buy_price = list(stock[stock['Shares'] == 1]['High'])[0]
    net_profit = df.loc[df.index[-2], 'wealth']
    buy_price = df['Close'].iloc[0]
    sell_price = df['Close'].iloc[0] + net_profit
    percentage = 100*(sell_price/buy_price)
    return percentage

#returns profit for Column L: Daily SS Strategy
def ss_wealth_return_2(df):
    df = add_lineK(df)
    df = add_lineD(df)
    df.dropna(inplace=True)

    initial_price = df.iloc[0]['Close']
    position = 0  # 0 means no position, 1 means holding stock
    final_price = initial_price

    for i in range(1, len(df)):
        # Buy condition
        if position == 0 and (df.iloc[i]['K'] - df.iloc[i]['D'] > 0.01) and (df.iloc[i]['K'] < 20):
            position = 1
            final_price = df.iloc[i]['Close']
        # Sell condition
        elif position == 1 and (df.iloc[i]['D'] - df.iloc[i]['K'] > 0.01) and (df.iloc[i]['K'] > 80):
            position = 0
            final_price = df.iloc[i]['Close']

    percentage_change = 100 * (final_price / initial_price)
    return percentage_change

def ss_wealth_return_20ma(df):
    df = add_lineK(df)
    df = add_lineD(df)
    df.dropna(inplace=True)

    # Calculate the 20-day moving average
    df['20_MA'] = df['Close'].rolling(window=20).mean()

    initial_price = df.iloc[0]['Close']
    position = 0  # 0 means no position, 1 means holding stock
    final_price = initial_price

    for i in range(1, len(df)):
        # Buy condition
        if position == 0 and (df.iloc[i]['K'] - df.iloc[i]['D'] > 0.01) and (df.iloc[i]['K'] < 20):
            position = 1
            final_price = df.iloc[i]['Close']
        # Additional buy condition: price touches 20-day MA
        elif position == 0 and df.iloc[i]['Close'] <= df.iloc[i]['20_MA']:
            position = 1
            final_price = df.iloc[i]['Close']
        # Sell condition
        elif position == 1 and (df.iloc[i]['D'] - df.iloc[i]['K'] > 0.01) and (df.iloc[i]['K'] > 80):
            position = 0
            final_price = df.iloc[i]['Close']

    percentage_change = 100 * (final_price / initial_price)
    return percentage_change

#returns profit for column M: SS daily AND only sell when P < 5MA
def ss_wealth_return_3(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df = add_lineK(df)
    df = add_lineD(df)
    df.dropna(inplace=True)

    initial_price = df.iloc[0]['Close']
    position = 0  # 0 means no position, 1 means holding stock
    final_price = initial_price

    for i in range(1, len(df)):
        # Buy condition
        if position == 0 and (df.iloc[i]['K'] - df.iloc[i]['D'] > 0.01) and (df.iloc[i]['K'] < 20):
            position = 1
            final_price = df.iloc[i]['Close']
        # Sell condition
        elif position == 1 and (df.iloc[i]['D'] - df.iloc[i]['K'] > 0.01) and (df.iloc[i]['K'] > 80) and (df.iloc[i]['Close'] < df.iloc[i]['MA5']):
            position = 0
            final_price = df.iloc[i]['Close']

    percentage_change = 100 * (final_price / initial_price)
    return percentage_change

#returns profit for column N: SS daily AND only sell when P < 5MA AND red candle
def ss_wealth_return_with_red_candle(df):
    df = add_lineK(df)
    df = add_lineD(df)
    df.dropna(inplace=True)

    # Calculate the 5-day moving average
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df.dropna(inplace=True)

    initial_price = df.iloc[0]['Close']
    position = 0  # 0 means no position, 1 means holding stock
    final_price = initial_price

    for i in range(1, len(df)):
        # Buy condition: Slow stochastic K line crosses above the D line and K < 20
        if position == 0 and (df.iloc[i]['K'] - df.iloc[i]['D'] > 0.01) and (df.iloc[i]['K'] < 20):
            position = 1
            buy_price = df.iloc[i]['Close']

        # Sell condition: Slow stochastic D line crosses above the K line and K > 80
        # AND the close price is below the 5-day moving average
        # AND the current price is showing a red candle (close price < open price and close price < previous close price)
        elif position == 1 and (df.iloc[i]['D'] - df.iloc[i]['K'] > 0.01) and (df.iloc[i]['K'] > 80) \
                and (df.iloc[i]['Close'] < df.iloc[i]['MA5']) \
                and (df.iloc[i]['Close'] < df.iloc[i]['Open']):
                #and (df.iloc[i]['Close'] < df.iloc[i-1]['Close'])
            position = 0
            sell_price = df.iloc[i]['Close']
            final_price = sell_price

    percentage_change = 100 * (final_price / initial_price)
    return percentage_change

#returns profit for Column K: 401k strategy, invest and do nothing
def buy_and_hold_return(df):
    # Calculate the initial buy price (first day's closing price)
    buy_price = df['Close'][0]

    # Calculate the final sell price (last day's closing price)
    sell_price = df['Close'][-1]

    # Calculate the net profit
    net_profit = 100*(sell_price/buy_price)

    # Calculate the percentage difference

    return net_profit

#returns profit for EMA strategy
def ema_wealth_return(df):
  df['MA10'] = df['Close'].rolling(10).mean()
  df['MA20'] = df['Close'].rolling(20).mean()
  df.dropna(inplace=True)

  initial_price = df.iloc[0]['Close']
  position = 0  # 0 means no position, 1 means holding stock
  final_price = initial_price

  for i in range(1, len(df)):
        # Buy condition
      if position == 0 and (df.iloc[i]['MA10'] > df.iloc[i]['MA20']):
          position = 1
          final_price = df.iloc[i]['Close']
        # Sell condition
      elif position == 1 and (df.iloc[i]['MA10'] < df.iloc[i]['MA20']):
          position = 0
          final_price = df.iloc[i]['Close']

  percentage_change = 100 * (final_price / initial_price)
  return percentage_change

#Use this to find the most recent slow stochastic crossover point for each etf in the list. Used just to test and make sure function is working.
def find_recent_slow_stochastic_cross(df):
    df = add_lineK(df)
    df = add_lineD(df)

    buy_cross_date = None
    sell_cross_date = None

    for i in range(1, len(df)):
        if (df.iloc[i]['K'] - df.iloc[i]['D'] > 0.01) and (df.iloc[i]['K'] < 20.0) and \
           (df.iloc[i-1]['K'] - df.iloc[i-1]['D'] <= 0.01 or df.iloc[i-1]['K'] >= 20.0):
            buy_cross_date = df.index[i]

        if (df.iloc[i]['D'] - df.iloc[i]['K'] > 0.01) and (df.iloc[i]['K'] > 80.0) and \
           (df.iloc[i-1]['D'] - df.iloc[i-1]['K'] <= 0.01 or df.iloc[i-1]['K'] <= 80.0):
            sell_cross_date = df.index[i]

    return buy_cross_date if buy_cross_date else sell_cross_date

def get_most_recent_cross(ticker):
    end = datetime.today()
    start = end - timedelta(days=365)
    interval = '1d'

    df = get_stock_data(ticker, start, end, interval)
    most_recent_cross_date = find_recent_slow_stochastic_cross(df)

    return most_recent_cross_date

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

today = date.today()
print("Today's date:", today)

daily_data = get_stock_data("PARA", today - timedelta(weeks=52),today, interval="1d")
daily_data = add_MA(daily_data,5)
daily_data = add_MA(daily_data,20)
daily_data = add_lineK(daily_data)
daily_data = add_lineD(daily_data)
#daily_data = wealth_return(daily_data)
graph(daily_data,"PARA")

##Main table function that uses helper functions to create a table of etfs with all the data needed to fill out the google sheets
def etf_table_ma_va(date, tickers):

  rows = []
  except_list = []


  for i in range(len(etf_list_2024)):
    etf_data = yf.Ticker(etf_list_2024[i])
    daily_data = get_stock_data(etf_list_2024[i], date - timedelta(weeks=52),date, interval="1d")
    daily_tenyear_data = get_tenyear_stock_data(etf_list_2024[i], date - relativedelta(years=10),date, interval="1d")
    weekly_data = get_tenyear_stock_data(etf_list_2024[i], date - relativedelta(years=10), date, interval="1wk")
    daily_data = add_MA(daily_data,5)
    daily_data = add_MA(daily_data,10)
    daily_data = add_MA(daily_data,20)
    daily_data = add_MA(daily_data,50)
    daily_data = add_average_volume(daily_data,5)
    daily_data = add_average_volume(daily_data,20)
    daily_data = add_lineK(daily_data)
    daily_data = add_lineD(daily_data)


    try:
  # company name
      company_name = etf_data.info['longName']

  # 52 Weeks Low
      min_index = daily_data['Low'].idxmin().date()
      min_price = round(daily_data['Low'].min(), 2)

  #52 Weeks High
      max_index = daily_data['High'].idxmax().date()
      max_price = round(daily_data['High'].max(), 2)

  #Current Price
      current_price = round(daily_data.iloc[-1]['Close'],2)

  #Position Ratio
      position_ratio = round((current_price - min_price)/(max_price - min_price)*100, 2)

  #Swing Size
      #swing_size = abs(relativedelta(min_index, max_index).months)

  # MA
      ma_5 = daily_data.iloc[-1]['MA5']
      # print(ma_5, "55555")
      ma_10 = daily_data.iloc[-1]['MA10']
      ma_20 = daily_data.iloc[-1]['MA20']
      # print(ma_20, "2020200")
      ma_50 = daily_data.iloc[-1]['MA50']
      # print(ma_50, "5050550")


  #average volume
      va_5 = daily_data.iloc[-1]['volume average5']
      # print(va_5, "dfnsjldfnsljfdnsdnfjk")
      va_20 = daily_data.iloc[-1]['volume average20']
      # print(va_20, "sjdfnksdnkfsjndfjn")

  #Kline , D line
      k = daily_data.iloc[-1]['K']
      # print(k, "k")
      d = daily_data.iloc[-1]['D']
      # print(d, "d")


  #wealth return (5MA strategy)
      fivema_profit = round(wealth_return(daily_tenyear_data), 2)


  #wealth return (slow stochastic daily)
      ss_daily_profit = round(ss_wealth_return_2(daily_tenyear_data), 2)

  #wealth return (SS + 20MA = current price)
      ss_daily_20ma_profit = round(ss_wealth_return_20ma(daily_tenyear_data), 2)

  #wealth return (SS + 5MA strategy)
      ss_daily_fivema_profit = round(ss_wealth_return_3(daily_tenyear_data), 2)

  #wealth return (SS + 5MA + red candle)
      ss_daily_red_candle_profit = round(ss_wealth_return_with_red_candle(daily_tenyear_data), 2)

  #wealth return (slow stochastic weekly)
      #ss_weekly_profit = round(ss_wealth_return_2(weekly_data), 2)

  #wealth return (401k strategy)
      buy_and_hold_profit = round(buy_and_hold_return(daily_tenyear_data), 2)

  #EMA return
      ema_profit = round(ema_wealth_return(daily_tenyear_data), 2)

  #most recent SS cross date
      cross_dates = get_most_recent_cross(tickers[i])


  #dividend amount
      #dividend = etf_data.info['dividendYield']
  #dividend date
      #dividend_date = etf_data.info['exDividendDate']

      new_row = [etf_list_2024[i], company_name, min_price,  max_price, current_price,
                position_ratio, ma_5, ma_10, ma_20,ma_50,va_5,va_20, k, d, buy_and_hold_profit, ss_daily_profit, ss_daily_20ma_profit, ss_daily_fivema_profit, ss_daily_red_candle_profit, ema_profit, fivema_profit, cross_dates]
      # print(new_row)
      rows.append(new_row)

    except:
      company_ticker = etf_list_2024[i]
      company_name = etf_names[i]
      print(company_name)
      except_list.append(company_name)


  columns = ['Stock', 'Name', '52 Weeks Low', '52 Weeks High','Current Price',
            'Position Ratio','ma_5','ma_10','ma_20','ma_50','va_5','va_20', 'k', 'd', '401k Profit', 'SS Daily Profit', 'SS Daily 20MA Profit' ,'SS Daily 5MA Profit', "SS Daily 5MA Red Candle Profit", 'EMA Profit', '5MA Profit', 'Most Recent Cross Date']

  df_before_screen = pd.DataFrame(rows, columns = columns)
  return df_before_screen

#Run the table function for a specific day on the etf list and sort the values alphabetically
date_str = "2024-07-02"
running_date = date.fromisoformat(date_str)

etf_2024_info = etf_table_ma_va(running_date, etf_list_2024).sort_values('Stock')
etf_2024_info

#Add in the 20 day slope column
etf_2024_info['20 days Slope'] = round(ta.roc(etf_2024_info['Current Price'], timeperiod=20).fillna(0), 2)
etf_2024_info

#Verifies that the function returns the right crossover dates and corresponds to Webull graphs
etf_2024_info.loc[etf_2024_info['Stock'] == 'SPY', 'Most Recent Cross Date']
