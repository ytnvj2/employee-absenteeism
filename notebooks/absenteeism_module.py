import pandas as pd,numpy as np, os, pickle as pkl
from sklearn.preprocessing import StandardScaler
class absenteeism():
    def __init__(self,model_file,scaler_file):
        self.model=pkl.load(open(model_file,'rb'))
        self.scaler=pkl.load(open(scaler_file,'rb'))
        self.data=None
    def load_and_process_data(self,data_path):
        df=pd.read_csv(data_path,sep=';')
        self.preprocessed_data=df.copy()
        cat_cols=['ID', 'Reason for absence', 'Month of absence', 'Day of the week','Seasons', 'Education','Son', 'Pet']
        num_cols=[ 'Transportation expense', 'Distance from Residence to Work','Service time', 'Age', 'Work load Average/day ', 'Hit target','Disciplinary failure', 'Social drinker','Social smoker', 'Weight', 'Height', 'Body mass index','Absenteeism time in hours']
        df.loc[df['Reason for absence'].isin(range(1,15)),'Reason for absence']=1
        df.loc[df['Reason for absence'].isin(range(15,19)),'Reason for absence']=2
        df.loc[df['Reason for absence'].isin(range(19,22)),'Reason for absence']=3
        df.loc[df['Reason for absence'].isin(range(22,29)),'Reason for absence']=4
        df.Education=df.Education.map({1:0,2:1,3:1,4:1})
        df.Pet=df.Pet.map({0:0,1:1,2:1,4:1,5:1,8:1})
        df.Son=df.Son.map({0:0,1:0,2:1,3:1,4:1})
        df=df[df['Month of absence']!=0]
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
        df.drop(['ID','Weight','Age','Social smoker','Disciplinary failure','Education','Pet'],axis=1)
        df=pd.get_dummies(df,columns=['Reason for absence'],drop_first=True)
        df=df[['Reason for absence_1', 'Reason for absence_2', 'Reason for absence_3', 'Reason for absence_4','Month of absence', 'Day of the week', 'Seasons','Transportation expense', 'Distance from Residence to Work', 'Service time', 'Work load Average/day ', 'Hit target', 'Son','Social drinker', 'Height', 'Body mass index']]
        self.data=df
        self.data.iloc[:,[7,8,9,10,11,14,15]]=self.scaler.transform(df.iloc[:,[7,8,9,10,11,14,15]])
    def predicted_probability(self):
        if(self.data is not None):
            pred=self.model.predict_proba(self.data)[:,1]
            return pred
    def predicted_class(self):
        if(self.data is not None):
            pred=self.model.predict(self.data)
            return pred
    def prediction_with_inputs(self):
        if(self.data is not None):
            self.preprocessed_data['Prediction']=self.model.predict(self.data)
            self.preprocessed_data['Prediction Prob']=self.model.predict_proba(self.data)[:,1]
            return pred
