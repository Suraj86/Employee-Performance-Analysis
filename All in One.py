import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
%matplotlib inline

# For Label Encoding
from sklearn.preprocessing import LabelEncoder

# For OverSampling
import scipy
from imblearn.combine import SMOTETomek

# For feature selection
from sklearn.ensemble import ExtraTreesRegressor

# For Scaling
from sklearn.preprocessing import StandardScaler

# For Splitting the dataset into training and testing
from sklearn.model_selection import train_test_split

# SVC Algorithm and XGBoost Algorithm
from sklearn.svm import SVC

# For testing the train data
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# For GridSearchCV and RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# For Error Rate
from sklearn import metrics

# For warnings
import warnings
warnings.filterwarnings('ignore')

# Reading the Dataset
df = pd.read_excel('INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls')

# Dropping the EmpNumber feature
df.drop(labels='EmpNumber', axis=1, inplace=True)

# Splitting the dataset into Independent and Dependent features
# Selecting all the columns except the dependent feature
X = df.iloc[:,:-1]
# Selecting the last dependent feature
y = df.iloc[:,-1]


# Initiaalising the LabelEncoder()
enc = LabelEncoder()
# Converting the categorical variables into discrete numerical variables
X['Gender'] = enc.fit_transform(X['Gender'])
X['EducationBackground'] = enc.fit_transform(X['EducationBackground'])
X['MaritalStatus'] = enc.fit_transform(X['MaritalStatus'])
X['EmpDepartment'] = enc.fit_transform(X['EmpDepartment'])
X['EmpJobRole'] = enc.fit_transform(X['EmpJobRole'])
X['BusinessTravelFrequency'] = enc.fit_transform(X['BusinessTravelFrequency'])
X['OverTime'] = enc.fit_transform(X['OverTime'])
X['Attrition'] = enc.fit_transform(X['Attrition'])

final_data = pd.DataFrame(data = X, columns=['EmpLastSalaryHikePercent','EmpEnvironmentSatisfaction',
                                             'YearsSinceLastPromotion','EmpDepartment',
                                             'ExperienceYearsInCurrentRole',
                                             'EmpWorkLifeBalance','YearsWithCurrManager','EmpJobRole','ExperienceYearsAtThisCompany'])

# Smote Oversampling Technique
smk = SMOTETomek(random_state=40)
X_res, y_res = smk.fit_sample(final_data,y)

# Initialising the Standard Scaler
scaler = StandardScaler()
# Fitting the data
df_scaled = scaler.fit_transform(X_res)

# Splitting data to train and test
X_train, X_test, y_train, y_test = train_test_split(df_scaled,y_res,test_size=0.20,random_state=0)

# Initialising SVM
model = SVC()
# Fitting the Model
model.fit(X_train, y_train)

# Initialising the values for HyperParameter Optimization
param_grid = {'C':[0.1,1,10,100,1000],
             'gamma':[1,0.1,0.01,0.001,0.0001],
             'kernel':['rbf']}

# Fitting the hyperparameter into SVM
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

# Fitting the model for grid search
grid.fit(X_train, y_train)

import pickle
file = open('SVM.pkl', 'wb')
pickle.dump(grid, file)

# Loading model to compare the results
model = pickle.load(open('SVM.pkl','rb'))
print(grid.predict([13,2,1,1,4,4,5,1,6]))