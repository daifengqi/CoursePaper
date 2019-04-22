library(readxl)    # 程序包：文件读取
library(lubridate) # 程序包：时间数据
library(stringr)   # 程序包：文本数据
library(limSolve)  # 程序包：解方程组

bond <- read_xlsx('D://R/TermStructure/CN_bond_19-04-15.xlsx')
bond <- bond[!is.na(bond$前收盘价),]
row.names(bond) <- 1:nrow(bond)
bond$付息日二[is.na(bond$付息日二)] <- bond$付息日一[is.na(bond$付息日二)]

# 第一次付息日期处理
pay_1st <- unlist(bond[,11])
pay_1st_month <- unlist(strsplit(str_extract(pay_1st, '(.|..)月'),'月'))
na_loc <- which(is.na(pay_1st_month))
pay_1st_month[na_loc] <- substr(pay_1st,5,6)[na_loc]
pay_1st_month <- as.numeric(pay_1st_month)

bond_in_1st <- pay_1st_month %in% c(4,7,10,1)
pay_1st_month <- pay_1st_month[bond_in_1st]
# 第二次付息日期处理
pay_2cd <- unlist(bond[,12])
pay_2cd_month <- unlist(strsplit(str_extract(pay_2cd, '(.|..)月'),'月'))
na_loc <- which(is.na(pay_2cd_month))
pay_2cd_month[na_loc] <- substr(pay_2cd,5,6)[na_loc]
pay_2cd_month <- as.numeric(pay_2cd_month)

bond_in_2cd <- pay_2cd_month %in% c(4,7,10,1)
pay_2cd_month <- pay_2cd_month[bond_in_2cd]


# 债券筛选
bond_in <- bond_in_1st + bond_in_2cd
bond_fit <- bond[as.logical(bond_in),]

bond_fit <- bond_fit[1:74,]

# 提取债券指标
# 到期年月
n <- 5
nmax <- 4*n  # 估计年限*4 = 估计季度
end_year <- year(bond_fit$到期日期)
end_year_real <- end_year
end_year[end_year > (2019+n)] <- 2019+n
end_month <- month(bond_fit$到期日期)

# 票面利率
coupon <- bond_fit$票面利率


# 变量补充函数
fill_col <- function(lst){
  if(length(lst) < nmax){
    n_fill <- nmax - length(lst)
    lst <- c(lst, rep(0, n_fill))
  }
  else{
    lst <- lst[1:nmax]
  }
  return(lst)
}


# 付息一次和付息两次的债券分开处理
eql <- which(pay_1st_month==pay_2cd_month)

# Start here  
df_fit <- c()
log_add <- c()
log_file <- c()
for(i in 1:nrow(bond_fit)){
  # 只付息一次
  if(i %in% eql){
    if(end_month[i]==1){
      real_paytime <- c(c(0,0,1), rep(c(0,0,0,1),end_year[i]-2020))
      fill_paytime <- fill_col(real_paytime)
      df_fit <- c(df_fit, fill_paytime)
      log_file <- c(log_file, '付息1次，1月到期')
    }
    else if(end_month[i]==4){
      real_paytime <- c(c(0,0,0,1), rep(c(0,0,0,1),end_year[i]-2020))
      fill_paytime <- fill_col(real_paytime)
      df_fit <- c(df_fit, fill_paytime)
      log_file <- c(log_file, '付息1次，4月到期')
    }
    else if(end_month[i]==7){
      real_paytime <- c(1, rep(c(0,0,0,1),end_year[i]-2019))
      fill_paytime <- fill_col(real_paytime)
      df_fit <- c(df_fit, fill_paytime)
      log_file <- c(log_file, '付息1次，7月到期')
    }
    else if(end_month[i]==10){
      real_paytime <- c(c(0,1),rep(c(0,0,0,1),end_year[i]-2019))
      fill_paytime <- fill_col(real_paytime)
      df_fit <- c(df_fit, fill_paytime)
      log_file <- c(log_file, '付息1次，10月到期')
    }
  }
  # 付息两次
  else{
    # 在2019年7或10月到期的
    if(end_year[i] == 2019){
      if(end_month[i] == 7){
        real_paytime <- c(1,0,0,0)
        fill_paytime <- fill_col(real_paytime)
        df_fit <- c(df_fit, fill_paytime)
        log_file <- c(log_file, '付息两次，2019年7月到期')
      }
      else if(end_month[i] == 10){
        real_paytime <- c(0,1,0,0)
        fill_paytime <- fill_col(real_paytime)
        df_fit <- c(df_fit, fill_paytime)
        log_file <- c(log_file, '付息两次，2019年10月到期')
      }
    }
    else{
      if(end_month[i]==1){
        real_paytime <- c(c(1,0,1), rep(c(0,1,0,1),end_year[i]-2020))
        fill_paytime <- fill_col(real_paytime)
        df_fit <- c(df_fit, fill_paytime)
        log_file <- c(log_file, '付息两次，2020年以后1月到期')
      }
      else if(end_month[i]==4){
        real_paytime <- rep(c(0,1,0,1),end_year[i]-2019)
        fill_paytime <- fill_col(real_paytime)
        df_fit <- c(df_fit, fill_paytime)
        log_file <- c(log_file, '付息两次，2020年以后4月到期')
      }
      else if(end_month[i]==7){
        real_paytime <- c(1, rep(c(0,1,0,1),end_year[i]-2019))
        fill_paytime <- fill_col(real_paytime)
        df_fit <- c(df_fit, fill_paytime)
        log_file <- c(log_file, '付息两次，2020年以后7月到期')
      }
      else if(end_month[i]==10){
        real_paytime <- c(c(0,1),rep(c(0,1,0,1),end_year[i]-2019))
        fill_paytime <- fill_col(real_paytime)
        df_fit <- c(df_fit, fill_paytime)
        log_file <- c(log_file, '付息两次，2020年以后10月到期')
      }
    }
  }
  l <- length(df_fit)
  log_add <- c(log_add,l)
}

df_fit <- matrix(df_fit, nrow = nmax)
df_fit <- t(df_fit)


# 填充现金流
# 现金流矩阵
cashflow <- df_fit
# 利息与到期现金加入现金流
for(i in 1:nrow(cashflow)){
  if(end_year[i] == end_year_real[i]){
  cashflow[i,] <- cashflow[i,] * coupon[i]
  cashflow[i,max(which((cashflow[i,] == coupon[i])))] <- 100 + coupon[i]
  }
  else{
  cashflow[i,] <- cashflow[i,] * coupon[i]
  }
}


# 回归
y <- bond_fit$前收盘价
y <- y[20:length(y)]
X <- cashflow
X <- X[20:nrow(X),]
m1 <-lm(y~0+X)
r_lm <- 1/m1$coefficients - 1
plot(r_lm)

# 受约束回归
A <- X
B <- y
N <- nmax
G_0 <- rbind(cbind(rep(0,N-1),-1*diag(N-1)),rep(0,N))
G_1 <- G_0 + diag(N)
G_2 <- diag(c(-1,rep(0,N-1)))
G <- rbind(G_2,G_1)
H <- c(-1,rep(0,N-1),rep(0, N))


reg <- lsei(A = A, B = B, G = G, H = H, type=2)
# 系数
reg
coe <- reg$X
r <- 1/coe-1
plot(r, type='l')

# 年化利率
# 年化利率
m <- seq(3,n*12,3)
ry <- r * 12/m
ts.plot(ry)
ry


