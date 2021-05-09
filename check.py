#!/usr/bin/env python
# coding: utf-8

# In[98]:


#reference: https://www.youtube.com/watch?v=QIUxPv5PJOY
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt



# In[99]:


company_list=["INTC", "AMD", "CSCO", "AAPL", "MU", "NVDA", "QCOM", "AMZN", "NFLX", "FB", "GOOG", "BABA", 
              "EBAY", "IBM", "XLNX", "TXN", "NOK", "TSLA", "MSFT", "SNPS"]


stock_result=[]
change_rate_list=[]
epoch=2
batch=4
#company=company_list[0]
for company_count in range(20):

    # In[100]:
    company=company_list[company_count]


    quote2=web.DataReader(company, data_source="yahoo", start="2021-04-30", end="2021-05-06")
    print(quote2["Close"])

    # In[121]:


    print(company)
    change_rate=float(100*(quote2["Close"][-1]-quote2["Close"][-2])/ quote2["Close"][-2] )
    print("change rate : {}%".format(change_rate))
    change_rate_list.append(change_rate)
    if change_rate>1.5 :
        stock_result.append(0)
    elif change_rate<-1.5:
        stock_result.append(2)
    else:
        stock_result.append(1)



for i in range(len(stock_result)):
    print(stock_result[i])

print("-------")

for i in range(len(change_rate_list)):
    print("{}: {}%".format(company_list[i],change_rate_list[i]))

        

    # In[ ]:


