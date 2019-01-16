# ׼���׶�
library(vars)
library(tseries)
library(ggplot2)
library(ggthemes)
setwd('D:/R')

# ��ȡ����
quartmac <- readxl::read_xlsx('�������new�����ȣ�.xlsx')
quausd <- readxl::read_xlsx('USDX(Q).xlsx')
macrodata <- quartmac[,c(1,3,4,6)]
colnames(macrodata) <- c('��������','Ͷ���γ�','���ڶ�','������')
np <- readxl::read_xlsx('profit2.xlsx')
df <- np[,-1]
col <- colorRampPalette(c("white", "grey",'black')) #������ɫ


## VARģ��
# ����׼��
dfmod <- as.data.frame(cbind(diff(df$�����), diff(df$ͨ��),diff(df$����),
                             quartmac$`ƽ������:��Ԫ�������`[2:64]), 
                       stringsAsfactors=F)
colnames(dfmod) <- c('Comp','Sign','Elec','Rate')
dfmod[,1:3] <- dfmod[,1:3]/1000000 # ͳһ��λ������

# ƽ���Լ���
adf.test(df$�����)
adf.test(diff(df$�����))
adf.test(df$ͨ��)
adf.test(diff(df$ͨ��))
adf.test(df$����)
adf.test(diff(df$����))
adf.test(dfmod$����)
adf.test(diff(dfmod$����))
# �ͺ����
VARselect(dfmod)
k <- min(VARselect(dfmod)$selection)
# ���VARģ��
varmod <- VAR(dfmod, lag.max = 2, type='both')
# �鿴ģ��
summary(varmod)
plot(varmod)
# ��λ������:�Ƿ�ƽ��
# ��AR��������ʽ��������м��飬����Ƿ����ڵ�λ��Բ��

roots(varmod, modulus = F) # ������
roots(varmod, modulus = T) # ��λԲ�ĵľ���
# ������Ӧ
var.irf<-irf(varmod)
plot(var.irf,names='Rate')

# ����ֽ�
depo <- fevd(varmod)
plot(depo)

depoex <- as.data.frame(depo$Rate)
### ����ʡ��
xlsx::write.xlsx(depoex,'����ֽ�.xlsx',row.names = T)
lev <- colnames(depoex)
# �Լ���ͼ
vec <- c() 
for(i in 1:10){
  for(j in 1:4){
    vec <- c(vec, i,lev[j],depoex[i,j])
  }
}
depodf <- as.data.frame(matrix(vec,nrow=40,ncol=3,byrow = T), stringsAsFactors = F)
depodf[,3] <- as.numeric(depodf$V3)
colnames(depodf)[3] <- 'Value'
# ���Լ���ͼ,��excel
ggplot(depodf,aes(V1,Value,fill=V2))+
  geom_bar(stat="identity",position="fill")+
  theme_wsj()+
  scale_fill_wsj("rgby", "")+
  theme(axis.ticks.length=unit(0.5,'cm'))+
  guides(fill=guide_legend(title=NULL))






