library(readxl)
library(imputeMissings)
library(DMwR2)
library(psych)
library(ggplot2)
library(corrgram)
library(car)
library(dummies)
library(randomForest)
library(caTools)
library(FNN)
library(DMwR)
library(rpart)
library(forecast)
library(vars)
df = read_excel('Absenteeism_at_work_Project.xls')
df$`Reason for absence`[df$`Reason for absence`==0]=20
df$`Month of absence`[df$`Month of absence`==0]=NA
cat_cols=c(1,2,3,4,5,13)
num_cols=c(6:11,14,17:21)
for (i in cat_cols) {
  df[,i]=as.factor(df[,i,drop=T])
}
summary(df)
# Missing Value Imputation
missing_val_imputer=function(df,num_cols,cat_cols){
  df_full=df[0,]
  for (i in unique(df$ID)) {
    df_n=df[df$ID==i,]
    df_n=impute(df_n)
    df_full=rbind(df_full,df_n)
  }
  return(df_full)
}

df=missing_val_imputer(df,num_cols,cat_cols)

# Lets step into visualization
#  For Numerical Variables, we will be using the multi.hist fucntion to visualize
#  All variables in one go. The plots will contain each variable's histogram,
#  KDE plot, and a line representing Normal Distribution for comparison.
#   Import package from library

multi.hist(df[,num_cols[1:6]],dcol =c('blue','red'), dlty = c('solid','solid'),main = 'Variable Analysis' )
multi.hist(df[,num_cols[-c(1:6)]],dcol =c('blue','red'), dlty = c('solid','solid'),main = 'Variable Analysis' )
#   Now for the factors, lets plot bar graph to see the count of each class 
init=ggplot(data=df)
for(i in cat_cols){
  plot=init+geom_bar(aes(x=df[,colnames(df[,i,drop=F])]),fill='blue',colour='blue')+xlab(colnames(df[,i,drop=F]))+
    ylab('Frequency')
  print(plot)
}
# Plots the boxplots for all the numerical variables
for(i in num_cols){
  plot=init+geom_boxplot(aes(y=df[,colnames(df[,i,drop=F])]))+ylab(colnames(df[,i,drop=F]))
  print(plot)
}

# Creates the histogram for the variable that were found to be skewed
# lh is the vector containing index of the skewed variables
lh <- c(6,8,9,10,11,17,19,21)
# bw is binwidth to be selected for each variable
bw <- c(40,3,6,20000,2, 0.8, 3.6,12)
for (i in 1:8) {
  plot <- init +
    geom_histogram(aes(x = df[,lh[i]]), binwidth = bw[i])+
    geom_vline(aes(xintercept = mean(df[,lh[i]])), color = "red")+
    geom_vline(aes(xintercept = median(df[,lh[i]])),color='blue')+
    xlab(colnames(df)[lh[i]])+
    ylab("Frequency")+
    ggtitle(paste("Histogram, Median, and Mean of ",colnames(df)[lh[i]], ""))
  print(plot)
}

# Creates the Histograms for the skewed Variables with Outliers
# lh is the vector containing index of the skewed variables
lh <- c(8,21)
# bw is binwidth to be selected for each variable
bw <- c(3,12)
for (i in 1:2) {
  print(         init+ 
                   geom_histogram(aes(x = df[,lh[i]]), binwidth = bw[i],color='black')+
                   xlab(colnames(df)[lh[i]])+
                   ggtitle(paste("Histogram Plot")))
}

outlierImputer=function(df,num_cols){
  while (TRUE) {
    tot_miss=NULL
    for(i in num_cols){
      val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
      df[,i][df[,i] %in% val]= NA
      tot_miss=c(tot_miss,length(val))
    }
    print(sum(tot_miss))
    if(sum(tot_miss)>0){
      df=knnImputation(df,k = 3)
    }
    else{
      break
    }
  }
  return(df)
}


df=outlierImputer(df,c(8,21))

#Feature Engineering
df$Total.Utilization=(df$Service.time-df$Absenteeism.time.in.hours)/df$Service.time
for (i in unique(df$Month.of.absence)){
  monthly_utilization=sum(df[df$Month.of.absence==i,22])/length(df[df$Month.of.absence==i,22])
  df[df$Month.of.absence==i,'Monthly.Utilization']=monthly_utilization
}

# feature selection
numFeatureSel=function(df,num_cols){
  corr=cor(df[,num_cols])
  print('Eigen Values for Correlation Matrix')
  print(eigen(corr)$values) # if values in decreasing order than multicollinearity present
  # Condition Number: max Eigen Value / min Eigen Value
  print('Condition Number')
  print(max(eigen(corr)$values)/min(eigen(corr)$values)) # if grater than 100, multicollinearity present
  corrgram(df[,num_cols],order = F,upper.panel = panel.pie,text.panel = panel.txt,main='Correlation Plot')
  return(num_cols[-c(4,11)])
}
numSel=numFeatureSel(df,c(num_cols))
numFeatureSel(df,num_cols[-c(4,11)])

catFeatureSel=function(df,cat_cols){
  cSel=NULL
  for(i in cat_cols){
    x=summary(aov(Absenteeism.time.in.hours~df[,i], data = df))
    print(x)
    if(x[[1]]$`Pr(>F)`[1]<0.05){
      cSel=c(cSel,i)
    }
  }
  return(cSel)
}
cSel=catFeatureSel(df,c(cat_cols,12,16,15))



df_o=df[,c(cSel[-1],numSel)]
df_o=dummy.data.frame(df_o, sep = "." )
df_o[,c(29,1)]=NULL

VIF_check=function(df_o){
  while (T) {
    lr_model=lm(Absenteeism.time.in.hours~.,data = df_o)
    x=vif(lr_model)
    maxVar = max(x)
    if (maxVar > 6){
      j = which(x == maxVar)
      df_o = df_o[, -j]
    }
    else{
      break()
    }
  }
  return(df_o)
}

df_vif=VIF_check(df_o)



backwardEliminationRF=function(df,sl){
  numVars = length(df)
  for (i in c(1:numVars)){
    imppred=randomForest(formula =Absenteeism.time.in.hours ~ ., data = df,ntree = 100, keep.forest = FALSE, importance = TRUE)
    minVar =min(importance(imppred, type = 1))
    if (minVar < sl){
      j = which(importance(imppred, type = 1) == minVar)
      df = df[, -j]
    }
    numVars = numVars - 1
  }
  return(imppred)
  
}

rf_model=backwardEliminationRF(df_o,5)
importance(rf_model, type = 1)




backwardElimination=function(df,sl){
  numVars = length(df)
  for (i in c(1:numVars)){
    regressor = lm(formula = Absenteeism.time.in.hours~., data = df)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      df = df[, -j]
    }
    numVars = numVars - 1
  }
  return(regressor)
  
}
#  Modeling the Data
# Divide data into train and test using stratified sampling method

set.seed(101) 
sample = sample.split(df_vif$Absenteeism.time.in.hours, SplitRatio = .80)
train = subset(df_vif, sample == TRUE)
test  = subset(df_vif, sample == FALSE)
lr_model=backwardElimination(df_vif,0.05)
summary(lr_model)
y_pred=predict(lr_model,test[,-c(1:3,7,11,15:16,22,26,28:36,38,40:44,46,47,48)])
#Summary of Linear model
summary(lr_model)
r=sum((y_pred-test[,48])^2)/sum((test[,48]-mean(test[,48]))^2)
1-r #-> 0.48
regr.eval(trues = test[,48],preds = y_pred,train.y = train[,48],stats = c('mae','mse','rmse','mape','nmse','nmae'))



convertToTimeSeries=function(df){
  ts=df[0,]
  cat_cols=c(2:5,12,13,14,15,16,17)
  num_cols=c(6:11,18:23)
  for(m in unique(df$Month.of.absence)){
    x=df[df$Month.of.absence==m,]
    n=x[0,num_cols]
    n=rbind(n,data.frame(as.list(colMeans(x[,num_cols]))))
    c=x[0,]
    for(i in cat_cols){
      m=names(table(x[,i]))[table(x[,i])==max(table(x[,i]))]
      if(length(m)>1){
        c[1,i]=m[1]
      }
      else{
        c[1,i]=m[1]
      }
    }
    c[1,num_cols]=n
    c[,c(12,14,15,16,17)]=as.numeric(c[,c(12,14,15,16,17)])
    c=c[,-1]
    ts=rbind(ts,c)
  }
  ts=ts[order(ts$Month.of.absence),]
  rownames(ts)=ts$Month.of.absence
  c[,cat_cols]=as.numeric(c[,cat_cols[-c(12,14,15,16,17)]])
  c[,cat_cols]=as.factor(c[,cat_cols[-c(12,14,15,16,17)]])
  ts$Month.of.absence=as.numeric(ts$Month.of.absence)
  ts$Total.Utilization=NULL
  
  return(ts)
}

ts=convertToTimeSeries(df)



ggplot(data = ts)+ geom_line(aes(x=Month.of.absence,y=Monthly.Utilization))

ggplot(data = ts)+ geom_line(aes(x=Month.of.absence,y=Absenteeism.time.in.hours))

ggplot(data = ts)+ geom_line(aes(x=Work.load.Average.day,y=Absenteeism.time.in.hours))

ggplot(data = ts)+ geom_line(aes(x=Work.load.Average.day,y=Monthly.Utilization))

ggplot(data = ts)+ geom_line(aes(x=Distance.from.Residence.to.Work,y=Absenteeism.time.in.hours))

ggplot(data = ts)+ geom_line(aes(x=Distance.from.Residence.to.Work,y=Monthly.Utilization))



ts_full=stats::ts(ts[,c(5,6,7,9,10,20)])

var=VAR(y=ts_full)


x=forecast(object = var,h=12)

ts_pred=data.frame(x$forecast)
ts_pred=ts_pred[,c(1,6,11,16,21,26)]
colnames(ts_pred)=colnames(ts[,c(5,6,7,9,10,20)])
ts_pred$Total.Utilization=(ts_pred$Service.time-ts_pred$Absenteeism.time.in.hours)/ts_pred$Service.time
print(paste('Total loss due to Absenteeism is ',(1-mean(ts_pred$Total.Utilization)),'%'))
plot(ts_pred$Absenteeism.time.in.hours,type='l',xlab='Month',ylab='Absenteeism')

