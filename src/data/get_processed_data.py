import os, numpy as np, pandas as pd
import fancyimpute
def read_data():
    raw_data_path=os.path.join(os.path.pardir,os.path.pardir,'data','raw','Absenteeism_at_work.csv')
    df=pd.read_csv(raw_data_path,sep=';')
    cat_cols=['ID', 'Reason for absence', 'Month of absence', 'Day of the week',
           'Seasons', 'Education','Son', 'Pet']
    num_cols=[ 'Transportation expense', 'Distance from Residence to Work',
           'Service time', 'Age', 'Work load Average/day ', 'Hit target',
           'Disciplinary failure', 'Social drinker',
           'Social smoker', 'Weight', 'Height', 'Body mass index',
           'Absenteeism time in hours']
    return df,cat_cols,num_cols
def process_data(df,num_cols):
    return (df.pipe(change_categories).
            pipe(outlier_imputer,num_cols=num_cols).
            drop(['ID','Weight','Age','Social smoker','Disciplinary failure','Education','Pet'],axis=1).
            pipe(pd.get_dummies,columns=['Reason for absence'],drop_first=True)
           )
def change_categories(df):
    df.loc[df['Reason for absence'].isin(range(1,15)),'Reason for absence']=1
    df.loc[df['Reason for absence'].isin(range(15,19)),'Reason for absence']=2
    df.loc[df['Reason for absence'].isin(range(19,22)),'Reason for absence']=3
    df.loc[df['Reason for absence'].isin(range(22,29)),'Reason for absence']=4
    df.Education=df.Education.map({1:0,2:1,3:1,4:1})
    df.Pet=df.Pet.map({0:0,1:1,2:1,4:1,5:1,8:1})
    df.Son=df.Son.map({0:0,1:0,2:1,3:1,4:1})
    df=df[df['Month of absence']!=0]
    return df
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
    return df_o
def write_data(df):
    df=df[['Reason for absence_1', 'Reason for absence_2', 'Reason for absence_3', 'Reason for absence_4',
           'Month of absence', 'Day of the week', 'Seasons','Transportation expense', 'Distance from Residence to Work',
           'Service time', 'Work load Average/day ', 'Hit target', 'Son','Social drinker', 'Height', 'Body mass index',
           'Absenteeism time in hours']]
    df.to_csv(os.path.join(os.path.pardir,os.path.pardir,'data','processed','absenteeism_processed.csv'))
print(__name__)
if __name__=='__main__':
    df,cat_cols,num_cols=read_data()
    df=process_data(df,num_cols)
    write_data(df)
