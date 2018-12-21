import os
import pandas as pd
import numpy as np
import statistics as stat
import fancyimpute
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def load_data():
    # import the dataset
    df=pd.read_excel('Absenteeism_at_work_Project.xls')
    # Convert the categorical columns to object 
    cat_cols=["ID", "Reason for absence", "Month of absence",
              "Day of the week", "Seasons","Disciplinary failure",
              "Education","Social drinker","Social smoker"]
    for i in cat_cols:
        df[i] = df[i].astype(object)
    num_cols=["Transportation expense","Distance from Residence to Work", "Service time","Age",
              "Work load Average/day ","Hit target","Son",
              "Pet", "Weight","Height",
              "Body mass index","Absenteeism time in hours" ]
    for i in num_cols:
        df[i] = df[i].astype(np.float64)
    # viewing the dataframe's info
    df.info()
    return df,num_cols,cat_cols
def impute_missing_vals(df,num_cols,cat_cols):
    df_full=pd.DataFrame(columns=df.columns)
    for j in np.unique(df.ID):
        df_n=df[df.ID==j].reset_index(drop=True)
        missing_val = df_n.isnull().sum()
        r,c=df_n.shape
        if r>2:
            if df_n[num_cols].isnull().sum().sum()>0:
                df_n[num_cols]=pd.DataFrame(fancyimpute.KNN(k = 5).complete(df_n[num_cols]), columns = num_cols)
#         for i in num_cols:
#             if len(df_n[df_n[i].isnull()])>0:
#                 df_n.loc[df_n[i].isnull(),i]=np.mean(df_n[i])
        for i in cat_cols:
            if len(df_n[df_n[i].isnull()])>0:
                if len(stat._counts(df_n.loc[:,i]))>1:
                    df_n.loc[:,i]=df_n.loc[:,i].fillna(method='ffill')
                    df_n[i] = df_n[i].astype(object)
                else:
                    df_n.loc[df_n[i].isnull(),i]=stat.mode(df_n[i])
        df_full=pd.concat([df_full,df_n],ignore_index=True)
    return df_full
def outlier_imputer(df_o,num_cols):
    # Outlier Analysis
    while True:
        for i in num_cols:
            median=np.median(df_o[i])
            std=np.std(df_o[i])
            min=(df_o[i].quantile(0.25)-1.5*(df_o[i].quantile(0.75)-df_o[i].quantile(0.25)))    
            max=(df_o[i].quantile(0.75)+1.5*(df_o[i].quantile(0.75)-df_o[i].quantile(0.25)))
            df_o.loc[df_o[i]<min,i] = np.nan
            df_o.loc[df_o[i]>max,i] = np.nan
        missing_val = df_o.isnull().sum()
        if(missing_val.sum()>0):
            df_o[num_cols]=pd.DataFrame(fancyimpute.KNN(k = 3).complete(df_o[num_cols]), columns = num_cols)
#             for i in num_cols:
#                 if len(df_o[df_o[i].isnull()])>0:
#                     df_o.loc[df_o[i].isnull(),i]=np.mean(df_o[i])
        else:
            break
    df_o.loc[df_o['Reason for absence']==0,'Reason for absence']=20
    df_o.loc[df_o['Month of absence']==0,'Month of absence']=stat.mode(df_o['Month of absence'])
    return df_o
def add_features(df):
    # Feature Engineering
    df['Time Utilization']=(df['Service time'] - df['Absenteeism time in hours'])/df['Service time']
    for i in df['Month of absence'].unique():
        monthly_utilization=df[df['Month of absence']==i].iloc[:,21].sum()/len(df[df['Month of absence']==i])
        if ('Monthly Utilization' in df.columns):
            df.loc[df['Month of absence']==i,'Monthly Utilization']=monthly_utilization
        else:
            df['Monthly Utilization']=np.where(df['Month of absence']==i,monthly_utilization,np.nan)
    return df
def convert_to_timeseries(df):
    ts_df=pd.DataFrame(columns=df.columns)
    cat_cols=['Day of the week', 'Disciplinary failure', 'Education','Reason for absence', 'Seasons', 'Social drinker','Social smoker']
    num_cols=["Transportation expense","Distance from Residence to Work", "Service time","Age",'Monthly Utilization',
                  "Work load Average/day ","Hit target","Son",
                  "Pet", "Weight","Height",
                  "Body mass index","Absenteeism time in hours" ]
    for i in df['Month of absence'].unique():
        x=df[df['Month of absence']==i]
        n=pd.DataFrame(np.mean(x[num_cols]).values.reshape(1,-1),columns=num_cols)
        c=pd.DataFrame(columns=cat_cols)
        for i in cat_cols:
            if len(stat._counts(x.loc[:,i]))>1:
                c.loc[0,i]=stat._counts(x.loc[:,i])[0][0]
            else:
                c.loc[0,i]=stat.mode(x[i])
        c['Month of absence']=np.mean(x['Month of absence'])
    #     s=pd.DataFrame(np.mean(df[df['Month of absence']==i]).values.reshape(1,-1),columns=df.columns)
        ts_df=pd.concat([ts_df,pd.concat([n,c],axis=1,sort=True)],ignore_index=True,sort=True)
    ts_df=ts_df.sort_values(by='Month of absence')
    ts_df.reset_index(drop=True,inplace=True)
    ts_df.drop(labels=['ID','Time Utilization'],axis=1,inplace=True)
    cat_cols=['Day of the week', 'Disciplinary failure', 'Education','Reason for absence', 'Seasons', 'Social drinker','Social smoker']
    for i in cat_cols:
    #     ts_df[i]=round(ts_df[i],0)
        ts_df[i]=ts_df[i].astype(object)
    ts=ts_df.set_index(keys='Month of absence')
    ts['Social smoker']=ts['Social smoker'].astype(np.float64)
    ts['Social drinker']=ts['Social drinker'].astype(np.float64)
    ts=pd.get_dummies(ts,drop_first=True)
#     print(ts.head())
    ts=ts.drop(columns=[
       'Day of the week_3', 'Day of the week_4', 'Day of the week_5',
       'Day of the week_6', 'Seasons_2', 'Seasons_3', 'Seasons_4'],axis=1)
    return ts
def feature_selection(df,num_cols):
    #Set the width and hieght of the plot
    f, ax = plt.subplots(figsize=(7, 5))
    #Generate correlation matrix
    corr = df[num_cols].corr()
    #Plot using seaborn library
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    return df
def train_VAR(df):
    endog=['Absenteeism time in hours', 'Distance from Residence to Work',
        'Hit target',  'Service time',
        'Transportation expense',  'Work load Average/day ']
    exog=['Height','Monthly Utilization', 'Pet','Son','Weight', 'Reason for absence_27.0', 'Reason for absence_28.0']
    #fit the model
    model = VAR(endog=df[endog])
    model_fit = model.fit()
    return model_fit
def forecast(train,model_fit):
    # make prediction on validation
    endog=['Absenteeism time in hours', 'Distance from Residence to Work',
        'Hit target',  'Service time',
        'Transportation expense',  'Work load Average/day ']
    prediction = model_fit.forecast(model_fit.y, steps=12)
    prediction=pd.DataFrame(prediction)
    prediction.columns=train[endog].columns
    prediction['Time Utilization']=(prediction['Service time'] - prediction['Absenteeism time in hours'])/prediction['Service time']
    print('Total loss due to absenteeism ',1-np.mean(prediction['Time Utilization']),'%')
    return prediction
def feature_scaling(X_train,X_test):
    standardScaler=StandardScaler()
    X_train[:,6:]=standardScaler.fit_transform(X_train[:,6:])
    X_test[:,6:]=standardScaler.transform(X_test[:,6:])
    return X_train,X_test,standardScaler
def train_lm(df):
    X=df.iloc[:,:20].values
    y=df.iloc[:,20].values
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123)
    X_train_scaled,X_test_scaled,standardScaler=feature_scaling(X_train,X_test)
    lr_model=LinearRegression()
    lr_model.fit(X_train,y_train)
    print('Linear Model Accuracy ',lr_model.score(X_test,y_test))
    preds=lr_model.predict(X_test)
    return pred
if __name__=='__main__':
    df,num_cols,cat_cols=load_data()
    df=impute_missing_vals(df,num_cols,cat_cols)
    df=outlier_imputer(df,["Service time","Work load Average/day ","Hit target","Absenteeism time in hours" ])
    df=add_features(df)
    ts=convert_to_timeseries(df)
    ts=feature_selection(ts,ts.columns)
    ts.drop(columns=["Body mass index","Age"],axis=1,inplace=True)
    ts=feature_selection(ts,ts.columns)
    model=train_VAR(ts)
    pred=forecast(ts,model)
    train_lm(df)