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
epoch=8
batch=2
predict_days=21
start_date="2014-01-01"
last_date="2021-05-06"
#company=company_list[0]
for company_count in range(20):

    # In[100]:
    company=company_list[company_count]

    df=web.DataReader(company, data_source="yahoo", start=start_date, end=last_date)
    df


    # In[101]:


    df.shape


    # In[102]:

    """
    plt.figure(figsize=(16,8))
    plt.title("CLose Price History")
    plt.plot(df["Close"])
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price USD ($)", fontsize=18)
    """

    # In[103]:


    data=df.filter(["Close"])
    dataset=data.values
    training_data_len=math.ceil(len(dataset)*0.8)
    training_data_len


    # In[104]:


    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(dataset)

    scaled_data


    # In[105]:


    train_data=scaled_data[0:training_data_len, :]
    x_train=[]
    y_train=[]
    for i in range(predict_days, len(train_data)):
        x_train.append(train_data[i-predict_days:i, 0])
        y_train.append(train_data[i,0])
        if i<=61:
            pass
            #print(x_train)
            #print(y_train)
            #print("-----------")


    # In[106]:


    x_train, y_train= np.array(x_train), np.array(y_train)


    # In[107]:


    x_train.shape


    # In[108]:


    x_train=np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape


    # In[109]:



    model=Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))


    # In[110]:


    model.compile(optimizer="adam", loss="mean_squared_error")


    # In[111]:


    model.fit(x_train, y_train, batch_size=batch, epochs=epoch)


    # In[112]:


    test_data=scaled_data[training_data_len-predict_days: , :]
    x_test=[]
    y_test=dataset[training_data_len: , :]
    for i in range(predict_days,len(test_data)):
        x_test.append(test_data[i-predict_days:i, 0])


    # In[113]:


    x_test =np.array(x_test)


    # In[114]:


    x_test=np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    # In[115]:


    predictions=model.predict(x_test)
    predictions=scaler.inverse_transform(predictions)


    # In[116]:


    rmse=np.sqrt(np.mean(predictions-y_test)**2)
    rmse


    # In[117]:


    train=data[:training_data_len]
    valid=data[training_data_len:]
    valid["Predictions"]=predictions
    """
    plt.figure(figsize=(16,8))
    plt.title("Model")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price USD ($)", fontsize=18)
    plt.plot(train["Close"])
    plt.plot(valid[["Close", "Predictions"]])
    plt.legend(["Train", "Valid", "Predictions"], loc="lower right")
    """


    # In[118]:


    valid


    # In[119]:


    quote=web.DataReader(company, data_source="yahoo", start=start_date, end=last_date)
    new_df=quote.filter(["Close"])

    last_predict_days=new_df[-predict_days:].values
    last_predict_days_scaled=scaler.transform(last_predict_days)
    X_test=[]
    X_test.append(last_predict_days_scaled)
    X_test=np.array(X_test)
    X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1], 1))
    pred_price=model.predict(X_test)
    pred_price=scaler.inverse_transform(pred_price)
    print(pred_price)


    # In[120]:


    quote2=web.DataReader(company, data_source="yahoo", start="2021-04-30", end=last_date)
    print(quote2["Close"])

    print("next day prediction: {}".format(pred_price))


    # In[121]:


    print(company)
    change_rate=float(100*(pred_price-quote2["Close"][-1])/ quote2["Close"][-1] )
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


