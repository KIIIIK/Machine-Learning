import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#生成线性可分的二分类数据
X, y = make_blobs(n_samples=1000, centers=2, n_features=2, 
                center_box=(0, 10), random_state=1111)
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'g^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.show()

#把标签值改为-1和1
y = np.where(y==0, -1, 1)

#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                    random_state=111)

plt.plot(X_train[:, 0][y_train == -1], X_train[:, 1][y_train == -1], 'g^')
plt.plot(X_train[:, 0][y_train == 1], X_train[:, 1][y_train == 1], 'bs')
plt.show()

class Perceptron():
    def __init__(self, learning_rate=1, cycle=10):
        """
        默认学习率为1,训练轮数为10
        """
        self.learning_rate = learning_rate
        self.cycle = cycle

    def activation(self, result):    
        """
        激活函数
        """
        if result >= 0:
            return 1
        else:
            return -1

    def train(self, X_train, y_train):
        #初始化参数
        self.w = np.array([[0], [0]])
        self.b = np.array([[0]])
        #获取自变量维度,如这里是2
        dim = X_train.shape[1]

        n = 0
        cycle = 1
        acc = 0

        #当训练集准确率大于0.99或者到达指定轮数时停止训练
        while acc < 0.99 and cycle <= self.cycle:
            print("******************")
            print("第%d轮: " % cycle)
            for x, y in zip(X_train, y_train):       
                if self.activation(np.dot(x, self.w) + self.b) == y:
                    continue
                
                #更新参数
                self.w = self.w + self.learning_rate * x.reshape(dim, 1) * y
                self.b = self.b + self.learning_rate * y
                
                n += 1
                print("---------------")
                print("第%d次迭代: " % n)
                print("参数w为: {}".format(self.w))
                print("参数b为: {}".format(self.b))
                print("---------------")
            
            print("******************")
            y_pre = np.dot(X_train, self.w) + self.b
            y_pre = np.where(y_pre>=0, 1, -1)
            acc = accuracy_score(y_train, y_pre.reshape(-1))
            print("第%d轮计算后, 训练集准确率为:%f" % (cycle, acc))
            cycle += 1
        
        # return w, b
    
    def predict(self, X_test):
        """
        进行预测
        """
        y_pre = np.dot(X_test, self.w) + self.b
        y_pre = np.where(y_pre>=0, 1, -1)
        # acc = accuracy_score(y_test, y_pre.reshape(-1))
        return y_pre

    def get_coef(self):
        """
        获取参数w
        """
        return self.w

    def get_intercept(self):
        """
        获取截距项b
        """
        return self.b        


clf = Perceptron()
## 开始训练
clf.train(X_train, y_train)
## 进行预测
y_pre = clf.predict(X_test)
## 计算准确率
acc = accuracy_score(y_test, y_pre.reshape(-1))
## 获取w参数
clf.get_coef()
## 获取截距项b
clf.get_intercept()

