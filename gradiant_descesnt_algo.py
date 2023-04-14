import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as Li_Reg

def model_():
    df = pd.read_csv("D:\\ML & DS\\CodeBasics\\py\\ML\\3_gradient_descent\\Exercise\\test_scores.csv")
    reg = Li_Reg()
    reg.fit(df[['math']],df.cs)
    return reg.coef_, reg.intercept_


def gradient_descent(x,y):
    mcurr = bcurr = 0
    n = len(x)
    iterations = 1000000
    lr = 0.0002
    for i in range(iterations):
        y_p = mcurr * x + bcurr
        md = -(2/n) * sum(x * (y - y_p))
        bd = -(2/n) * sum(y - y_p)
        mcurr = mcurr - lr * md
        bcurr = bcurr - lr * bd

    return mcurr, bcurr


df = pd.read_csv("D:\\ML & DS\\CodeBasics\\py\\ML\\3_gradient_descent\\Exercise\\test_scores.csv")
x = np.array(df.math)
y = np.array(df.cs)

m_my, b_my = gradient_descent(x,y)
m_sk, b_sk = model_()

print('m_my = {}, b_my = {}'.format(m_my, b_my))
print('m_sk = {}, b_sk = {}'.format(m_sk, b_sk))