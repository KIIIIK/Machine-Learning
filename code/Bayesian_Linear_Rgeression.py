import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#生成10个自变量的回归数据,并返回对应的参数
X, y, coef = make_regression(n_samples=1000, n_features=10, coef=True,
                                random_state=111)

#划分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                random_state=111)


class BayesianLinearRgeression():
    def __init__(self):
        return None
    
    def train(self, X_train, y_train, alpha=1, beta=1):
        """
        alpha: 参数w先验分布的协方差系数, 即w的协方差矩阵为(1/alpha)*I, 
                I为单位矩阵, 即w~N(0, (1/alpha)*I), 以此来简化计算, 默认为1
        beta: 噪声ε的精度系数, 即ε~N(0, 1/ε), 默认为1
        """        
        #在数据面前添加一列1, 简化计算, 随之计算出来的第一个w参数为截距项
        X = np.insert(X_train, 0, 1, axis=1)
        #返回训练数据的维度
        N, P = X.shape[0], X.shape[1]
        Y = y_train.reshape(N, 1)
        #计算w后验分布的精度矩阵
        precision_matrix = alpha * np.eye(P) + beta * np.dot(X.T, X)
        #计算w后验分布的协方差矩阵
        self.convariance = np.linalg.inv(precision_matrix)
        temp1 = np.dot(self.convariance, X.T)
        #计算w后验分布的均值
        self.mean = beta * np.dot(temp1, Y)
        #从w后验分布中采样出一组参数, 因为是随机采样, 每次采样出来的参数都不相同
        self.w = np.random.multivariate_normal(self.mean.reshape(-1), 
                                        self.convariance, size=1)       
    
    def predict(self, X_test):
        X = np.insert(X_test, 0, 1, axis=1)
        y_pred = np.dot(X, self.mean)

        return y_pred
            
    def get_coef(self):
        """
        获取参数w
        """
        return self.w[0][1:]

    def get_intercept(self):
        """
        获取截距项b
        """
        return self.w[0][0]      


if __name__ == "__main__":
    reg = BayesianLinearRgeression()
    reg.train(X_train, y_train)
    y_pre = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pre)
    reg.get_coef()
    reg.get_intercept()














