import pandas  as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
df = pd.read_csv(r'',  sep='\t',
engine='python', delimiter=',')

###Clean Null values

#Drop Null Values
bcdf = df.dropna(axis = 0, how ='any')

print(bcdf.columns)
#Convert Columns we are interested in to categorical variables
bcdf.REPORTING_AREA = bcdf['REPORTING_AREA'].astype('category')
bcdf.DISTRICT = bcdf['DISTRICT'].astype('category')
bcdf.UCR_PART = bcdf['UCR_PART'].astype('category')
bcdf.OFFENSE_CODE = bcdf['OFFENSE_CODE'].astype('category')
print(bcdf.dtypes)
#No need to normalise as we are using a decision tree

#One Hot Encode the categorical variables (cheating because we are pulling in the offense code)
test = pd.get_dummies(bcdf, columns=['OFFENSE_CODE','DISTRICT', 'UCR_PART'], drop_first=True)
#Select numeric feature values
feature_cols = [ 'YEAR', 'MONTH', 'HOUR']

#Splice Columns
x = pd.concat([test.loc[:,feature_cols],test.iloc[:, 18: ]],axis=1) #Combine numeric and categorical feature values
y = test.iloc[:,2] #Predicting the offense code

#Split into training and testing (67% of the data wll be test data)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state= 56)

#Build a decision tree using the entropy index
clf_en = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
                                  max_depth=3, min_samples_leaf=5)
clf_en.fit(x_train, y_train)

#Prediction on x_test based on model
y_pred = clf_en.predict(x_test)
#Print out score - 75% if using offence codes but only 62% when offence codes are removed
print(accuracy_score(y_test,y_pred))
#
#
