import pandas as pd, numpy as np,os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,roc_curve
from sklearn.externals import joblib
def load_data():
    processed_data_path=os.path.join(os.path.pardir,os.path.pardir,'data','processed')
    processed_df_path=os.path.join(processed_data_path,'absenteeism_processed.csv')
    df=pd.read_csv(processed_df_path,index_col=0)
    return df
def convert_to_class(df):
    df['Absenteeism Class']=np.where(df['Absenteeism time in hours']>df['Absenteeism time in hours'].median(),1,0)
    df.drop(['Absenteeism time in hours'],axis=1,inplace=True)
    return df
def scale_data(df):
    num_col=['Transportation expense', 'Distance from Residence to Work',
                       'Service time', 'Work load Average/day ', 'Hit target','Height', 'Body mass index']
    scaler=StandardScaler()
    df_scaled=df.copy()
    df_scaled[num_col]=scaler.fit_transform(df_scaled[num_col])
    return df_scaled,scaler
def split_data(df):
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    return train_test_split(X,y,test_size=0.2,random_state=0)
def train_model(X_train,y_train):
    logit=LogisticRegression()
    logit.fit(X_train,y_train)
    return logit
def model_preformance(model,X_train,X_test,y_train,y_test):
    print('Accuracy for Train Data',accuracy_score(y_train,model.predict(X_train)))
    print('Accuracy for Test Data',accuracy_score(y_test,model.predict(X_test)))
    print('Precision Score',precision_score(y_test,model.predict(X_test)))
    print('Recall Score',recall_score(y_test,model.predict(X_test)))
    print('Confusion Matrix',confusion_matrix(y_test,model.predict(X_test)))
def model_summary(model,df):
    summary=pd.DataFrame(columns=df.columns[:-1],data=model.coef_)
    summary=summary.reset_index()
    summary.columns=['Intercept', 'Reason for absence_1', 'Reason for absence_2',
           'Reason for absence_3', 'Reason for absence_4', 'Month of absence',
           'Day of the week', 'Seasons', 'Transportation expense',
           'Distance from Residence to Work', 'Service time',
           'Work load Average/day ', 'Hit target', 'Son', 'Social drinker',
           'Height', 'Body mass index']
    summary.Intercept=model.intercept_
    s=summary.T
    s.columns=['Coefficient']
    s.sort_values('Coefficient')
    s['Odds Ratio']=np.exp(s['Coefficient'])
    print(s)
def persist_model(model,scaler):
    joblib.dump(model,r'model.pickle')
    joblib.dump(scaler,r'scaler.pickle')

if __name__=='__main__':
    df=load_data()
    df=convert_to_class(df)
    df,scaler=scale_data(df)
    X_train,X_test,y_train,y_test=split_data(df)
    model=train_model(X_train,y_train)
    model_summary(model,df)
    model_preformance(model,X_train,X_test,y_train,y_test)
    persist_model(model,scaler)
