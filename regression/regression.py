# Shouts: sentdex

import pandas as pd
import quandl
import math

df = quandl.get('WIKI/GOOGL')
# print(df.columns)

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_CHANGE'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_CHANGE', 'Adj. Volume']]

# print(df.head())

# Cannot use Adjusted Close as a label since HL_PCT and PCT_CHANGE depend on the closing values
# We could predict the future close price

fc_col = 'Adj. Close'

df.fillna(-99999, inplace=True)

# Trying to forecast 1% of df. 
fc_out = int(math.ceil(0.01*len(df)))

# After shifting, the label of each row will be 10% into the future
df['label'] = df[fc_col].shift(-fc_out)
df.dropna(inplace=True)

print(df.head())
# print(df.tail())
# print(df.describe())
# print(df.size)

