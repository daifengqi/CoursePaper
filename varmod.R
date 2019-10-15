# 准备阶段
library(vars)
library(tseries)
library(ggplot2)
library(ggthemes)
# setwd('D:/R')

# 读取数据
quartmac <- readxl::read_xlsx('宏观数据new（季度）.xlsx')
quausd <- readxl::read_xlsx('USDX(Q).xlsx')
macrodata <- quartmac[,c(1,3,4,6)]
colnames(macrodata) <- c('最终消费','投资形成','出口额','净出口')
np <- readxl::read_xlsx('profit2.xlsx')
df <- np[,-1]
col <- colorRampPalette(c("white", "grey",'black')) #设置颜色


## VAR模型
# 数据准备
dfmod <- as.data.frame(cbind(diff(df$计算机), diff(df$通信),diff(df$电子),
                             quartmac$`平均汇率:美元兑人民币`[2:64]), 
                       stringsAsfactors=F)
colnames(dfmod) <- c('Comp','Sign','Elec','Rate')
dfmod[,1:3] <- dfmod[,1:3]/1000000 # 统一单位：百万

# 平稳性检验
adf.test(df$计算机)
adf.test(diff(df$计算机))
adf.test(df$通信)
adf.test(diff(df$通信))
adf.test(df$电子)
adf.test(diff(df$电子))
adf.test(dfmod$汇率)
adf.test(diff(dfmod$汇率))
# 滞后阶数
VARselect(dfmod)
k <- min(VARselect(dfmod)$selection)
# 拟合VAR模型
varmod <- VAR(dfmod, lag.max = 2, type='both')
# 查看模型
summary(varmod)
plot(varmod)
# 单位根检验:是否平稳
# 对AR特征多项式的逆根进行检验，逆根是否都落在单位复圆内

roots(varmod, modulus = F) # 复向量
roots(varmod, modulus = T) # 单位圆心的距离
# 脉冲响应
var.irf<-irf(varmod)
plot(var.irf,names='Rate')

# 方差分解
depo <- fevd(varmod)
plot(depo)

depoex <- as.data.frame(depo$Rate)
### 后面省略
xlsx::write.xlsx(depoex,'方差分解.xlsx',row.names = T)
lev <- colnames(depoex)
# 自己作图
vec <- c() 
for(i in 1:10){
  for(j in 1:4){
    vec <- c(vec, i,lev[j],depoex[i,j])
  }
}
depodf <- as.data.frame(matrix(vec,nrow=40,ncol=3,byrow = T), stringsAsFactors = F)
depodf[,3] <- as.numeric(depodf$V3)
colnames(depodf)[3] <- 'Value'
# 不自己作图,用excel
ggplot(depodf,aes(V1,Value,fill=V2))+
  geom_bar(stat="identity",position="fill")+
  theme_wsj()+
  scale_fill_wsj("rgby", "")+
  theme(axis.ticks.length=unit(0.5,'cm'))+
  guides(fill=guide_legend(title=NULL))







