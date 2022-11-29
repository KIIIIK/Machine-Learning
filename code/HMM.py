import numpy as np

def Viterbi(A, B, pi, O, dict_):
    """
    A : 状态转移矩阵
    B : 发射矩阵
    pi : 初始状态分布
    O : 观测序列
    dict_ : 观测值对应的字典
    """    
    #先初始化
    init = pi * B[:, dict_[O[0]]]

    delta = []
    fi = []
    n = len(O)
    for i in range(1, n):
        #计算由前一时刻状态转移到当前时刻状态的概率
        temp = init[:, np.newaxis] * A
        #计算当前时刻状态出现当前观测值的概率
        temp = temp * B[:, dict_[O[i]]]
        #计算最大概率
        temp_delta = np.max(temp, axis=0)
        temp_fi = np.argmax(temp, axis=0)
        init = temp_delta
        delta.append(temp_delta)
        fi.append(temp_fi)

    T = []
    for i in range(n-2, -1, -1):       
        #从最后时刻开始回溯
        if i == n-2:
            temp_T = np.argmax(delta[-1])
            T.append(temp_T)
        else:
            temp_T = fi[i][temp_T]
        T.append(temp_T)
    
    T.reverse()
    return T

if __name__ == "__main__":
    A = np.array([[0.5, 0.2, 0.3],
                [0.3, 0.5, 0.2],
                [0.2, 0.3, 0.5]])

    B = np.array([[0.5, 0.5],
                [0.4, 0.6],
                [0.7, 0.3]])

    pi = np.array([0.2, 0.4, 0.4])

    dict_ = {'红' : 0, '白' : 1}
    O = ['红', '白', '红', '白']

    T = Viterbi(A, B, pi, O, dict_)

