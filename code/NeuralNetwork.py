import numpy as np
# import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy.linalg import LinAlgError

#生成5个自变量的回归数据,并返回对应的参数
X, y, coef = make_regression(n_samples=1000, n_features=10, coef=True,
                                random_state=121)

#划分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                random_state=121)


















