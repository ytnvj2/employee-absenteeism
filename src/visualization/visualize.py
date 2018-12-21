import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
def bar_plots(df,cat_cols):
    for i in cat_cols:
        df[i].value_counts().plot(kind='bar')
        plt.xlabel(i)
        plt.show()
def histograms(df,num_cols):
    for i in num_cols:
        df[i].plot(kind='hist')
        plt.xlabel(i)
        plt.show()
def boxplots(df,cat_cols,num_cols):
    for i in num_cols:
        df[i].plot(kind='box')
        plt.show()
    for i in cat_cols:
        sns.boxplot(x=df[i],y=df['Absenteeism time in hours'])
        plt.show()
def scatterplots(df,num_cols):
    for i in num_cols:
        sns.scatterplot(x=df[i],y=df['Absenteeism time in hours'])
        plt.show()
def correlation_plot(df,num_cols):
    f, ax = plt.subplots(figsize=(7, 5))
    #Generate correlation matrix
    corr = df[num_cols].corr()
    #Plot using seaborn library
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True, ax=ax)
    plt.show()
if __name__=='__main__':
    df,cat_cols,num_cols=read_data()
    bar_plots(df,cat_cols)
    histograms(df,num_cols)
    boxplots(df,cat_cols,num_cols)
    scatterplots(df,num_cols)
    correlation_plot(df,num_cols)
