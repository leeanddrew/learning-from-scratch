import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    advertising=pd.read_csv("../datasets/Advertising.csv",index_col=0)

    x=advertising['TV'].values
    y=advertising['sales'].values

    x_bar = x.mean()
    y_bar = y.mean()

    b_1=np.sum(((x-x_bar)*(y-y_bar)))/np.sum(((x-x_bar)**2))
    b_0=y_bar-(b_1*x_bar)

    y_hat=b_0+b_1*x

    RSS=np.sum((y-y_hat)**2)
    TSS=np.sum((y-y_bar)**2)

    R_squared=1-(RSS/TSS)
    print("----------------------------------------")
    print("b_0 from scratch:",b_0)
    print("b_1 from scratch:",b_1)
    print("R_squared from scratch:",R_squared)

    print("----------------------------------------")

    x=advertising['TV'].values.reshape(-1,1)
    lin_reg=LinearRegression()
    lin_reg.fit(x,y)
    lin_reg.score(x,y)

    print("b_0 from sklearn:",lin_reg.intercept_)
    print("b_1 from sklearn:",lin_reg.coef_)
    print("R_squared from sklearn:",lin_reg.score(x,y))
    print("----------------------------------------")