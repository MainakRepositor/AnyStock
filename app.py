import warnings
warnings.filterwarnings('ignore')  # Hide warnings
import datetime as dt
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import os

from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import streamlit as st


#Importing Libraries done

#title
st.title('Stock Market App')
'---------------------------------------------------------'
#text
st.write("Developed by Mainak Chaudhuri")

image = Image.open(os.path.join('STOCK.jpg'))
st.image(image)

com = st.text_input("Enter the Stock Code of company","AAPL")

'You Enterted the company code: ', com

st_date= st.text_input("Enter Starting date as YYYY-MM-DD", "2000-01-10")

'You Enterted the starting date: ', st_date

end_date= st.text_input("Enter Ending date as YYYY-MM-DD", "2000-01-20")

'You Enterted the ending date: ', end_date

df = web.DataReader(com, 'yahoo', st_date, end_date)  # Collects data
df.reset_index(inplace=True)
df.set_index("Date", inplace=True)

#title
st.title('Stock Market Data')


df

'1. The Stock Open Values over time: '
st.line_chart(df["Open"])

'2. The Stock Close Values over time: '
st.line_chart(df["Close"])

#title
st.title('Moving Averages')
'---------------------------------------------------------'


mov_avg= st.text_input("Enter number of days Moving Average:", "50")

'You Enterted the Moving Average: ', mov_avg


df["mov_avg_close"] = df['Close'].rolling(window=int(mov_avg),min_periods=0).mean()

'1. Plot of Stock Closing Value for '+ mov_avg+ " Days of Moving Average"
'   Actual Closing Value also Present'
st.line_chart(df[["mov_avg_close","Close"]])

df["mov_avg_open"] = df['Open'].rolling(window=int(mov_avg),min_periods=0).mean()

'2. Plot of Stock Open Value for '+ mov_avg+ " Days of Moving Average"
'   Actual Opening Value also Present'
st.line_chart(df[["mov_avg_open","Open"]])




