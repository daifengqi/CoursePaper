library(readxl)    # 程序包：文件读取
library(lubridate) # 程序包：时间数据
library(stringr)   # 程序包：文本数据
library(limSolve)  # 程序包：解方程组

bond <- read_xlsx('D://R/TermStructure/CN_bond_19-04-15.xlsx')
bond <- bond[!is.na(bond$前收盘价),]
row.names(bond) <- 1:nrow(bond)

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


# 提取债券指标
# 到期年月
n <- 5
nmax <- 4*n
end_year <- year(bond_fit$到期日期)
end_year[end_year > (2019+n)] <- 2019+n
end_month <- month(bond_fit$到期日期)
end_time <- end_year + 1/12 * end_month
# 债券期限
duration <- bond_fit$债券期限
# 票面利率
coupon <- bond_fit$票面利率

# 自变量
fill_col <- function(lst){
  if(length(lst) < nmax){
    n_fill <- nmax - length(lst)
    lst <- c(lst, rep(0, n_fill))
  }
  return(lst)
}

# Start here  
df_fit <- c()
for(i in 1:nrow(bond_fit)){
  if(duration[i] <= 1){
    if(pay_1st_month[i] == 7) df_fit <- c(df_fit, fill_col(c(1,0,0,0)))
    else if(pay_1st_month[i] == 10) df_fit <- c(df_fit, fill_col(c(0,1,0,0)))
    else if(pay_1st_month[i] == 1) df_fit <- c(df_fit, fill_col(c(0,0,1,0)))
    else if(pay_1st_month[i] == 4) df_fit <- c(df_fit, fill_col(c(0,0,0,1)))
    else print('Warning: ...[0]')
  }
  else{
    if(pay_1st_month[i] == pay_2cd_month[i]){
      if(pay_1st_month[i] == 7) df_fit <- c(df_fit, fill_col(rep(c(1,0,0,0), end_year[i]-2019)))
      else if(pay_1st_month[i] == 10) df_fit <- c(df_fit, fill_col(rep(c(0,1,0,0), end_year[i]-2019)))
      else if(pay_1st_month[i] == 1) df_fit <- c(df_fit, fill_col(rep(c(0,0,1,0), end_year[i]-2019)))
      else if(pay_1st_month[i] == 4) df_fit <- c(df_fit, fill_col(rep(c(0,0,0,1), end_year[i]-2019)))
      else print('Warning: ...[1]')
    }
    else{
      if(pay_1st_month[i] == 7 && pay_2cd_month[i] == 1) df_fit <- c(df_fit, fill_col(rep(c(1,0,1,0), end_year[i]-2019)))
      else if(pay_1st_month[i] == 10 && pay_2cd_month[i] == 4) df_fit <- c(df_fit, fill_col(rep(c(0,1,0,1), end_year[i]-2019)))
      else if(pay_1st_month[i] == 1 && pay_2cd_month[i] == 7) df_fit <- c(df_fit, fill_col(rep(c(1,0,1,0), end_year[i]-2019)))
      else if(pay_1st_month[i] == 4 && pay_2cd_month[i] == 10) df_fit <- c(df_fit, fill_col(rep(c(0,1,0,1), end_year[i]-2019)))
      else print('Warning: ...[2]')
    }
  }
}


df_fit <- matrix(df_fit, nrow = nmax)
df_fit <- t(df_fit)

# 现金流矩阵
cashflow <- df_fit
# 利息与到期现金加入现金流
for(i in 1:nrow(cashflow)){
  cashflow[i,] <- cashflow[i,] * coupon[i]
  cashflow[i,max(which((cashflow[i,] == coupon[i])))] <- 100 + coupon[i]
}

# 受约束回归
y <- bond_fit$前收盘价
X <- cashflow
A <- X
B <- y
N <- nmax
G_0 <- rbind(cbind(rep(0,N-1),-1*diag(N-1)),rep(0,N))

G <- G_0 + diag(N)
H <- rep(0, N)

reg <- lsei(A = A, B = B, G = G, H = H, type=2)
# 系数
coe <- reg$X
r <- 1/coe-1
# ts.plot(r)
# r
# 年化利率
m <- seq(3,60,3)
ry <- r * 12/m
ts.plot(ry)
ry