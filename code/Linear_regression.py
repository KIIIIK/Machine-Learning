import numpy as np
# import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy.linalg import LinAlgError

#生成5个自变量的回归数据,并返回对应的参数
X, y, coef = make_regression(n_samples=1000, n_features=5, coef=True,
                                random_state=111)

#划分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                random_state=111)


class LinearRegression():
    def __init__(self):
        return None
    
    def train(self, X_train, y_train, method='OLS', 
                learning_rate=0.1, penalty=0.1):
        """
        method: 'OLS'(普通最小二乘回归), 'Ridge'(岭回归), 'SGD'(随机梯度下降)
        lerning_rate: 学习率, 默认为0.1, 只有在SGD方法下才会使用
        penalty: 正则化系数, 默认为0.1, 只有在Ridge方法下才会使用
        """        
        #在数据面前添加一列1, 简化计算, 随之计算出来的第一个w参数为截距项
        X = np.insert(X_train, 0, 1, axis=1)
        #返回训练数据的维度
        N, P = X.shape[0], X.shape[1]
        Y = y_train.reshape(N, 1)
        
        if method == 'OLS':
            try:
                temp1 = np.dot(X.T, X)
                temp2 = np.linalg.inv(temp1)
                temp3 = np.dot(temp2, X.T)
                self.w = np.dot(temp3, Y)
            except LinAlgError:
                #计算X的伪逆
                temp1 = np.linalg.pinv(X)
                self.w = np.dot(temp1, Y)
        elif method == 'Ridge':
            temp1 = np.dot(X.T, X)
            temp2 = temp1 + penalty * np.eye(P)
            temp3 = np.linalg.inv(temp2)
            temp4 = np.dot(temp3, X.T)
            self.w = np.dot(temp4, Y)    
        elif method == 'SGD':
            #初始化参数
            self.w = np.zeros((P, 1))
            mse = mean_squared_error(Y, np.dot(X, self.w))
            n = 1
            while mse > 0.1 and n <= 100:        
                #生成随机数
                random = np.random.randint(low=0, high=N, size=1)
                temp1 = Y[random] - np.dot(X[random], self.w)
                #更新参数
                self.w = self.w + learning_rate * temp1 * X[random].T
                #用更新后的参数进行预测
                y_pred = np.dot(X, self.w)
                #计算均方误差
                mse = mean_squared_error(Y, y_pred)
                print("^_^|^_^|^_^|^_^|^_^")
                print("第%d次迭代: " %n)
                print("参数w为: \n {}".format(self.w))
                print("此时训练集MSE={}".format(mse))
                n += 1

    def predict(self, X_test):
        """
        进行预测
        """
        X = np.insert(X_test, 0, 1, axis=1)
        y_pred = np.dot(X, self.w)

        return y_pred
    
    def get_coef(self):
        """
        获取参数w
        """
        return self.w[1:]

    def get_intercept(self):
        """
        获取截距项b
        """
        return self.w[0]             

if __name__ == "__main__":
    reg = LinearRegression()
    reg.train(X_train, y_train, method='SGD')
    y_pre = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pre)
    reg.get_coef()
    reg.get_intercept()

