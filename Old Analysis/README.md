# Employee Absenteeism
The objective of this Case Study is to analyze and forecast the Employee Absenteeism trend using the multivariate Time Series Data

## Tools Used:
##### R
##### Python
##### Jupyter Notebook
##### RStudio
##### Git

## R Packages:
##### readxl: Used for importing data from Excel
##### imputeMissings: Used for knnImputation
##### psych: Used for visualization
##### ggplot2: Used for visualization


##### DMWR: Same as DMWR2
##### corrgram: Used to create correlation plot
##### randomForest: Used to model Random Forest
##### rpart: Used to model Decision Trees
##### caTools: Used to partition the data in train and test
##### FNN: used for KNN regression
##### dummies: used to create dummy variables
##### car: used to calculate VIF values


## Python libraries
##### sklearn
##### pandas
##### numpy
##### os
##### fancyimpute
##### seaborn

## Information about the data:
The given data is multivariate time series data.
The details of data attributes in the dataset are as follows -
##### Individual identification (ID)
##### Reason for absence (ICD).
Absences attested by the International Code of Diseases (ICD) stratified into 21
categories (I to XXI) And 7 categories without (ICD)
##### Month of absence
##### Day of the week (Monday (2), Tuesday (3), Wednesday (4), Thursday (5), Friday (6))
##### Seasons (summer (1), autumn (2), winter (3), spring (4))
##### Transportation expense
##### Distance from Residence to Work (kilometers)
##### Service time
##### Age
##### Work load Average/day
##### Hit target
##### Disciplinary failure (yes=1; no=0)
##### Education (high school (1), graduate (2), postgraduate (3), master and doctor (4))
##### Son (number of children)
##### Social drinker (yes=1; no=0)
##### Social smoker (yes=1; no=0)
##### Pet (number of pet)
##### Weight
##### Height
##### Body mass index
##### Absenteeism time in hours (target)

## Problem Statement
The questions to be answered after analysis:
##### What changes company should bring to reduce the number of absenteeism?
##### How much losses every month can we project in 2011 if same trend of absenteeism continues?

## Findings of EDA:
The data consists of observations from 36 unique individuals and their absenteeism details and reason. The data contains missing values and so these will be imputed first. The best method to impute missing values will be by dividing the data based on the ID and imputing missing values in the filtered data using KNN Imputation. The rationale behind this is that we have multiple observations for each ID, so when rows of same ID are missing some values, then imputing with the help of these filtered observations will yeild a better and closer to actual value. After this, we analyze the data further and find that outliers are present in the data. For numerical variables, Outliers detected using boxplot in Transportation Expense, Service Time, Age, Work load Average/ day, Hit Target, Pet, Height, and Absenteeism time in hours. After looking more carefully at these outliers in the variables, we found that only Service Time & Absenteeism time in hours contain unrealistic values all other outliers are pretty possible so we only treat variables containing the unrealistic values. For this, the outliers are first assigned NA and then treated as missing values and KNN Imputation is used impute them. For Categorical variables, outliers found in Reason for absence and Month of absence where values have class which does not exist i.e. Month has no 0 month and Reason for absence has no 0 class. This implies that these values are outliers and need to be imputed. To impute these, we simply replace all the 0 values with mode of the Month of absence attribute and the Reason for absence is missing the value 20, this means the class 0 has been misclassified as 0 so we will be replacing 0 to 20. This completes Outlier Treatment. Correlation present in numerical variables between temp and atemp, so temp was removed. ANOVA test conducted to find association and identify significant predictors. VIF to reduce dimension after converting categorical to dummy variables. Random Forest importance metric used to reduce the variables further, these variables used for RF Model. To further help our analysis, we create a new feature named as Total Utilization. This new feature is calculated as the difference in Service time and Absenteeism time whole divided by the Service Time. This gives us an idea of the time utilization and thus help in guessing the loss. Feature Scaling performed on all variables except the target. This ends the Exploratory Data Analysis and We are now ready to step into Model Development.

## Models used:
##### Linear Regression: Used backward elimination to find the best model
##### Decision Tree Regression: Used ANOVA method to construct the Decision Tree.
##### Random Forest: Forest consisted of 97 trees, on plotting the error rate decreased with increase in no. of estimators.
##### KNN Regression: K was chosen as 11 after analyzing the test error and select k for which test error is the least.
##### Vector Auto Regression: Used to forecast the absenteeism time. The data was converted to time series containing 1 observation for each month which was calculated by averaging the attributes for each month.

## Model Evaluation:
##### R2 Values for Test Set for all models
###### Linear Model: 0.318
###### Decision Tree Model: 0.27
###### Random Forest Model: 0.212
###### KNN Model: 0.317
###### VAR Model: 0.532 but adjusted R2 was -0.17
##### Linear Regression resulted in the best adjusted R^2 value and hence was chosen as the best model.

## Conclusion:
After thoroughly analyzing the data, we come to the conclusion that absenteeism is dependent on the following variables:
##### Distance from Residence to Work
##### Transportation Expense
So the changes company should bring to reduce the absenteeism time is that they can provide transportation to the employees with longer distances at reduced prices.
Whereas when we forecast the loss to occur, if the same trend of absenteeism continues, then the overall loss of utilization will be approx. 34% time wasted in absenteeism.
