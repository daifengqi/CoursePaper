# Markov Chain
library(quantmod)        # 程序包：股票量化

# 获取贵州茅台数据
setSymbolLookup(MT=list(name='600519.ss',src='yahoo',from="2018-01-01", to='2019-01-01'))
getSymbols("MT")                       # 获取数据
chartSeries(MT, theme = 'white')       # 作K线图
mt <- as.vector(MT$`600519.SS.Close`)  # 提取数据

# 获得状态数据 
dmt <- diff(mt)           # 差分
thr <- sqrt(var(dmt))/3   # 标准差的1/3
sta <- rep(2,length(dmt)) # 初始化状态
sta[dmt>thr] <- 3
sta[dmt< (-thr)] <- 1
sta <- as.factor(sta)     # 得到三状态时间序列
summary(sta)

# 构造马尔科夫链——测试
library(markovchain)      # 程序包：Markov Chain
mk <- createSequenceMatrix(sta, sanitize = F)
adj <- diag(c(1/sum(mk[1,]),1/sum(mk[2,]),1/sum(mk[3,])))
mkc <- adj%*%mk

# 滚动预测(定义函数：markov_stock_predict_model)
markov_stock_predict_model <- function(sta, period){
right_cnt <- 0
for(i in 1:(length(sta)-period-1)){
time_start <- i
time_end <- time_start+period
mk <- createSequenceMatrix(sta[time_start:time_end], sanitize = F)
adj <- diag(c(1/sum(mk[1,]),1/sum(mk[2,]),1/sum(mk[3,])))
mkc <- adj%*%mk
end_state <- as.numeric(sta[time_end])
# 随机选取器
u <- c(0,0,0)
u[end_state] <- 1
unew <- u%*%mkc
umax <- unew==max(unew)
max_state <- c(1,2,3)[umax]
rv <- runif(1,0,length(max_state))
seed <- ceiling(rv)
new_state <- max_state[seed]
# 判断是否相等
rlt <- as.numeric(new_state == as.numeric(sta[(time_end+1)]))
if(as.logical(rlt)) right_cnt = right_cnt+1
}
rate <- right_cnt/(length(sta)-period-1)
return(rate)
}

# 准确率作图
t <- 30:210
rlist <- c()
for(i in 30:210){
r <- markov_stock_predict_model(sta, period=i)
rlist <- c(rlist,r)
}

plot(rlist~t, type='l',xlab='样本时间区间长度',ylab='预测准确率')
t[rlist == max(rlist)]
mean(rlist)
