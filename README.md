Employee Absenteeism Analysis
==============================

The objective of this Case Study is to analyze and forecast the Employee Absenteeism trend using the multivariate Time Series Data

## Tools Used:
##### Driven Data Cookie Cutter Template
##### Python
##### Jupyter Notebook
##### Git

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
The data consists of observations from 36 unique individuals and their absenteeism details and reason. The data does not contain missing values. We analyze the data by constructing bar plots for the categorical variables and histograms for the continuous variables. 
The bar plot reveals that Reason for absence has value 0 which means that there was no reason given and Month of absence contains 0 value for 3 obserations so remove these observations from our dataset. 
Moreover, Reason for absence is a nominal qualitative variable with 28 categories, we can group these 28 categories into 4 main categories as reasons 1-14 are common diseases, 15-18 are pregnancy related issues, 19-21 are related to poisoning and death, and 22-28 are appointment related. 0 being none of the reasons. So lets convert the 28 category variable into 5 categories. 
In case of Education, we see that majority of the observations are 1 and other observation can be grouped together. Similarly for Pets most of the individuals have 0 pets so we can convert the values greater than 0 to represent having a pet and 0 to not having a pet. For Son, we see that 2 or more children are rarely present so we can represent 2 or more children with 1 and less 2 children with 0. and find that outliers are present in the data. 
For numerical variables, Outliers detected using boxplot in Transportation Expense, Service Time, Age, Work load Average/ day, Hit Target, Pet, Height, and Absenteeism time in hours. These outliers are first assigned NA and then treated as missing values and KNN Imputation is used impute them. This completes Outlier Treatment. 
Correlation present in numerical variables. As weight and body mass index are correlated and Service time and Age are correlated we can remove one of each. Let's remove Weight and Age. Moreover, as Disciplinary Failure and Social Smoker are constant we can remove them as well. 
Boxplots constructed for categorical variables, and we see that boxplots of Pet and Education wrt Absenteeism are same for all the categories so we can drop Education and Pet from the data as well.
Now we convert the nominal categorical variable Reason for absence to dummy variables. The numerical features are then scaled using standard scaler. This ends the EDA.
As our problem looks like a regression problem as Absenteeism time is continuous, we can make it a classification problem by converting the Absenteeism time to Absenteeism class denoting excessive absenteeism whenever the absenteeism time greater than the mean.
To further help our analysis, we create a new feature named as Total Utilization. This new feature is calculated as the difference in Service time and Absenteeism time whole divided by the Service Time. This gives us an idea of the time utilization and thus help in guessing the loss. Feature Scaling performed on all variables except the target. This ends the Exploratory Data Analysis and We are now ready to step into Model Development.

## Models used:
##### Logestic Regression: Used backward elimination to find the best model
##### KNN Regression: K was chosen as 11 after analyzing the test error and select k for which test error is the least.

## Model Evaluation:
##### R2 Values for Test Set for all models
###### Logistic Model: 0.771 ( For Test Data)
###### KNN Regression Model: 0.187
###### KNN Classification Model: 0.722
##### Logistic Regression resulted in the best accuracy value and hence was chosen as the best model.

## Conclusion:
After thoroughly analyzing the data, we come to the conclusion that absenteeism is dependent on the following variables:
##### Transportation Expense
##### Distance from Residence to Work
So the changes company should bring to reduce the absenteeism time is that they can provide transportation to the employees with longer distances at reduced prices.
Whereas when we forecast the loss to occur, if the same trend of absenteeism continues, then the overall loss of utilization will be approx. 47% time wasted in absenteeism.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
