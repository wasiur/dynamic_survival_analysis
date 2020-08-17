setwd('D:\\GPSDE\\Data')
#This code converts the SIR data (generated from Wasiur's SIR_CLT paper) into a format into csv file, which makes it possible for the HMC to use.
for (stub1 in c('10','35','50')){
for (stub2 in c('05','10','15')){
for (stub3 in c('05','10','15')){
S0<-read.table(paste0('CM_sDataPoisson_',stub1,'_Exponential_',stub2,'_Exponential_',stub3,'_nS1000_nI10.txt'),sep=',')
I0<-read.table(paste0('CM_iDataPoisson_',stub1,'_Exponential_',stub2,'_Exponential_',stub3,'_nS1000_nI10.txt'),sep=',')
R0<-read.table(paste0('CM_rDataPoisson_',stub1,'_Exponential_',stub2,'_Exponential_',stub3,'_nS1000_nI10.txt'),sep=',')

n_individual<-10000 #N, you need to set a population size in order to convert the data.

S_obs<-round( colMeans(S0)*n_individual )
I_obs<-round( colMeans(I0)*n_individual )
R_obs<-round( colMeans(R0)*n_individual )

N_c=length(S_obs)
date=seq(as.Date("01-01-20",format='%d-%m-%y'), as.Date("01-01-20",format='%d-%m-%y') + N_c -1, "days")
date=format(date,format='%d-%m-%y')
I_delta<-c(0)
R_delta<-c(0)
for(i in 2:length(I_obs)){
  I_delta[i]<-I_obs[i]-I_obs[i-1]
  R_delta[i]<-R_obs[i]-R_obs[i-1]
}
#cum_confirm=I_obs+R_obs#+S_obs #Actually I am not sure it is good to include susceptible counts.
daily_confirm=I_delta+R_delta
#Daily confirm is the sum of I_delta (infected change in that day) and the R_delta (recovery change in that day)
cum_confirm<-c(0)
for(i in 2:length(daily_confirm)){
  cum_confirm[i]<-cum_confirm[i-1]+daily_confirm[i]
}
recovery=R_obs
time=as.character(date)
ret=cbind(1:length(time),time,cum_confirm,recovery,daily_confirm)
ret=as.data.frame(ret)
colnames(ret)<-c('','time','cum_confirm','recovery','daily_confirm')

write.csv(ret,paste0('HMC_CM_DataPoisson_',stub1,'_Exponential_',stub2,'_Exponential_',stub3,'_nS1000_nI10.csv'),row.names = F,quote=F)
}
}
}