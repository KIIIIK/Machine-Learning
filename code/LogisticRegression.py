import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.datasets import make_classification

#生成线性可分的二分类数据
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, 
                center_box=(0, 10), random_state=114)
# X, y = make_classification(n_samples=1000)
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'g^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.show()

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                    random_state=111)

plt.plot(X_train[:, 0][y_train == 0], X_train[:, 1][y_train == 0], 'g^')
plt.plot(X_train[:, 0][y_train == 1], X_train[:, 1][y_train == 1], 'bs')
plt.show()

class LogisticRegression():
    def __init__(self):
        return
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, X_train, y_train, learning_rate=0.1):
        X = np.insert(X_train, 0, 1, axis=1)
        N, P = X.shape
        #初始化参数
        # w = np.random.normal(0, 1, size=P)
        self.w = np.zeros((P, 1))
        n = 1
        acc = 0
        while acc < 0.99 and n <= 200:
            random = np.random.randint(0, N)
            x, y = X[random].reshape(P, 1), y_train[random]
            y_hat = self.sigmoid(self.w.T @ x)
            #计算梯度
            grad = (y_hat - y) * x
            #用随机梯度下降法更新参数
            self.w = self.w - learning_rate * grad
            #下面这两步是用牛顿法更新梯度
            # hessian = y_hat * (1 - y_hat) * (x @ x.T)
            # w = w - np.linalg.solve(hessian, grad)        
            #注意, 这里计算出来的p1是y=1的概率
            p1 = self.sigmoid(X @ self.w)
            temp = np.concatenate((1-p1, p1), axis=1)
            y_pre = np.argmax(temp, axis=1)
            # y_pre = np.where(p0 > 0.5, 0, 1)
            acc = accuracy_score(y_train, y_pre)
            print("^_^|^_^|^_^|^_^|^_^")
            print("第%d次迭代: " %n)
            print("参数w为: \n {}".format(self.w))
            print("此时训练集准确率={}".format(acc))        
            n += 1

    def predict(self, X_test):
        """
        进行预测
        """
        X = np.insert(X_test, 0, 1, axis=1)
        p1 = self.sigmoid(X @ self.w)
        temp = np.concatenate((1-p1, p1), axis=1)
        y_pred = np.argmax(temp, axis=1)        

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
    clf = LogisticRegression()
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    clf.get_coef()
    clf.get_intercept()


















