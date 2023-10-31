import numpy as np 
import pandas as pd

train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")

def missingdata(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    ms = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    ms = ms[ms["Percent"] > 0]
    return ms


print(missingdata(train_df))
print(missingdata(test_df))

# Drop Cabin, too much missing value
train_df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId']
              , axis=1, inplace = True)
test_df.drop(['Cabin', 'Name', 'Ticket'], 
             axis=1, inplace = True)

# Fill missing data
train_df['Age'].fillna(train_df['Age'].median(), inplace = True)
test_df['Age'].fillna(test_df['Age'].median(), inplace = True)

train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace = True)

# Feature engineering
all_data = [train_df,test_df]
for dataset in all_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


print(train_df.columns)
print(test_df.columns)

NUM_COLS = ['SibSp', 'Parch', 'Fare', 'FamilySize', 'Age']
CAT_COLS = ['Pclass', 'Sex', 'Embarked']

def get_train_data():
    return train_df.iloc[:, 1:], train_df['Survived']

def get_test_data():
    return test_df.iloc[:,1:], test_df['PassengerId']
