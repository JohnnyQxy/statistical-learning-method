import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from LinearRegression.linear_regression import LinearRegression


if __name__=='__main__':

    data_path='./data/'

    df_train=pd.read_csv(
        data_path+'train.csv',
        engine='python',
        encoding='utf-8'
    )

    df_test=pd.read_csv(
        data_path+'test.csv',
        engine='python',
        encoding='utf-8'
    )

    print(df_train.info())

    df_train=df_train[df_train['observation']=='PM2.5']
    df_test=df_test[df_test['AMB_TEMP']=='PM2.5']

    #删除无关特征
    df_train=df_train.drop(['Date','stations','observation'],axis=1)
    test_x=df_test.iloc[:,2:]

    train_x=[]
    train_y=[]

    for i in range(15):
        x=df_train.iloc[:,i:i+9]
        x.columns=np.array(range(9))

        y=df_train.iloc[:,i+9]
        y.columns=np.array(range(1))

        train_x.append(x)
        train_y.append(y)

    train_x=pd.concat(train_x)
    train_y=pd.concat(train_y)

    #将str数据类型转化为numpy的ndarray类型
    train_x=np.array(train_x,float)
    train_y=np.array(train_y,float)
    print(train_x)

    test_x=np.array(test_x,float)

    #数据归一化
    ss=StandardScaler()
    ss.fit(train_x)
    train_x=ss.transform(train_x)

    ss.fit(test_x)
    test_x=ss.transform(test_x)

    LR=LinearRegression()
    LR.fit_gd(train_x,train_y)

    #预测
    result=LR.predict(test_x)
    print(result)