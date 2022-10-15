import yfinance as yf 
import pandas as pd
import streamlit as st

st.write("""Stock Analysis""")

#stock symbol
ticker_symbol = yf.Ticker("MSFT")

stock_data = ticker_symbol.history(period='1m',start='2012-6-30', end='2022-6-30' )

st.line_chart(stock_data.Close)

dividend_data = ticker_symbol.dividends

dividend_data.to_csv(f'msftDividends.csv')
holders_data = ticker_symbol.major_holders
holders_data.to_csv(f'msftHolders.csv')