delta = 0.5
N = 500
x = numeric(N) #生成500个0

x[1] = 0
set.seed(2018-06-04)

for (i in 2:N){
  #print(i)
  eps = runif(1, -delta, delta) #从均匀分布(-0.5,0.5)中采样一个数
  y = x[i-1] + eps
}
