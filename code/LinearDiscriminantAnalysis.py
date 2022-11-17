import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#生成线性可分的二分类数据
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, 
                center_box=(0, 10), random_state=114)
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'g^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.show()

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                    random_state=111)

plt.plot(X_train[:, 0][y_train == 0], X_train[:, 1][y_train == 0], 'g^')
plt.plot(X_train[:, 0][y_train == 1], X_train[:, 1][y_train == 1], 'bs')
plt.show()

class LDA():
    def __init__(self):
        return None
    
    def train(self, X_train, y_train):
        #先把数据按标签分成两类
        C1 = X_train[np.argwhere(y_train==0).reshape(-1)]
        C2 = X_train[np.argwhere(y_train==1).reshape(-1)]
        #分别计算每类的均值
        m1 = np.mean(C1, axis=0)
        m2 = np.mean(C2, axis=0)
        # S1 = np.dot((C1 - m1).T, (C1 - m1))
        # S2 = np.dot((C2 - m2).T, (C2 - m2))
        #计算类内协方差矩阵
        cov_inclass = np.cov(C1, rowvar=False) + np.cov(C2, rowvar=False)   
        # S_w_inv = np.linalg.pinv(cov_inclass)
        # w = np.dot(S_w_inv, (m2 - m1).T)
        #计算参数w
        self.w = np.linalg.solve(cov_inclass, m2 - m1)
        #clip方法是为了保证范数不为0, 并归一化
        self.w = self.w / np.linalg.norm(self.w).clip(min=1e-10)     
        #计算各类投影的均值, 标量
        mu1 = np.dot(self.w, m1)
        mu2 = np.dot(self.w, m2)
        #计算threshold的链接:
        # https://stats.stackexchange.com/questions/4942/threshold-for-fisher-linear-classifier
        self.threshold = (mu1 + mu2) / 2

    def predict(self, X_test):
        #@代表进行矩阵乘法
        temp1 = X_test @ self.w.T
        y_pre = np.where(temp1 < self.threshold, 0, 1)

        return y_pre
    
    def get_coef(self):
        """
        获取参数w
        """
        return self.w

    def get_intercept(self):
        """
        获取threshold
        """
        return self.threshold  

if __name__ == "__main__":
    clf = LDA()
    clf.train(X_train, y_train)
    y_pre = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pre)




