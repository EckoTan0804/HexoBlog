---
title: Data Processing with Python
tags:
- Python
- Machine Learning
- Data Processing

---

This article based on the notes of [Machine Learning](https://his.anthropomatik.kit.edu/28_1008.php)'s exercise course from [Karlsruhe Institute of Technology](http://www.kit.edu/english/index.php). 

<!--more-->



## Preparation

+ IDE/ Notebook/ Editor: [JupiterLab](https://blog.jupyter.org/jupyterlab-is-ready-for-users-5a6f039b8906), [Jupyter Notebook](http://jupyter.org/)

+ Tools:
  + Math: [numpy](http://www.numpy.org/)
  + Data processing: [pandas](https://pandas.pydata.org/)
  + Plotting: [matplotlib.pyplot](https://matplotlib.org/api/pyplot_api.html),  [seaborn](https://seaborn.pydata.org/index.html)
  + Machine learning: [sklearn](http://scikit-learn.org/stable/)

```python
import pandas as pd
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
```

+ Dataset: We'll be using the **Titanic data set**. It contains the passenger data from the RMS Titanic, including whether a passenger survived the sinking of the ship or not. The dataset can be downloaded from [Kaggle](https://www.kaggle.com/c/3136/download-all).



## Read Datasets

`pd.read_csv()`

```python
# Load the train and test datasets from the CSV files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Combine the datasets for training and testing to one full data set
full_data = [train, test]
```

### Display Rows of Dataset

`head()`

```python
# Display the first 5 rows of the train data set
train.head(10)
```

###  Display Information of Dataset

```python
# Print the columns of the data frame
train.columns.values
```

```python
# Inspect the data, *info* can be used to show how complete or incomplete the
# dataset is
train.info()
```

To display information about a specific passenger, we can select a row with the following command:

```python
# iloc: index location
train.iloc[14]
```

```python
# Retrieve n number of samples from the data set
train.sample(5)
```

```python
# Retrieve a statistical description of the data set
train.describe()
```



## Visualizing the Data

We will use  [*seaborn*](https://seaborn.pydata.org/), a wrapper which uses matplotlib, but offers a higher-level interface for visualizing data.

### Draw a Figure with a Set of Subplots

[`plt.subplot`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplots.html): Create a figure and a set of subplots.

[`sns.coungplot`](https://seaborn.pydata.org/generated/seaborn.countplot.html#seaborn.countplot): Show the counts of observations in each categorical bin using bars.

[`sns.displot`](https://seaborn.pydata.org/generated/seaborn.distplot.html#seaborn.distplot): Flexibly plot a univariate distribution of observations.

[`sns.barplot`](https://seaborn.pydata.org/generated/seaborn.barplot.html): Show point estimates and confidence intervals as rectangular bars.

```python
f,ax = plt.subplots(1,4,figsize=(20,5))

sns.countplot('Pclass',hue='Survived',data=train,ax=ax[0],palette='husl')
ax[0].set_title('Survival Rate by Class')

sns.countplot('Sex',hue='Survived',data=train,ax=ax[1],palette='husl')
ax[1].set_title('Survival Rate by Sex')

sns.countplot('Embarked',hue='Survived',data=train,ax=ax[2],palette='husl')
ax[2].set_title('Survival Rate by Embarked')

sns.countplot('Survived', hue='Sex', data=train, ax=ax[3], palette='husl')
```

![output_25_1.png](https://github.com/EckoTan0804/HexoBlog/blob/Data-Processing-With-Py/source/images/Exercise_01_Data_Processing_with_Python/output_25_1.png?raw=true)



## Cleaning the Data

Data from the real world is messy. Normally there are missing values, outliers and invalid data (e.g. negative values for age) in a data set. We can solve problems with data quality by replacing these values, trying to close the gap by interpolation or by dropping the respective entries.

### Detecting and Filtering Outliers

Outliers that are either very large or small skew the overall view of the data. 

[`sns.regplot`](https://seaborn.pydata.org/generated/seaborn.regplot.html#seaborn.regplot): Plot data and a linear regression model fit.

```python
f,ax = plt.subplots(1,2,figsize=(13,5))
# The outliers in Fare (Fare paid by the passenger)
sns.regplot(x=train["PassengerId"], y=train["Fare"], fit_reg=False,ax=ax[0])
# SibSp(Number of siblings and spouses of the passenger aboard)
sns.regplot(x=train["PassengerId"], y=train["SibSp"], fit_reg=False, ax=ax[1])

ax[0].set_title('Total Passengers by Fare')
ax[1].set_title('Total Passengers by Number of siblings and spouses')

plt.show()
```

![output_30_0.png](https://github.com/EckoTan0804/HexoBlog/blob/Data-Processing-With-Py/source/images/Exercise_01_Data_Processing_with_Python/output_30_0.png?raw=true)

One way of detecting outliers could be the use of the standard deviation. If we assume that the data is normally distributed, then 95 percent of the data is within 1.96 standard deviations of the mean. So we can drop the values either above or below that range.

```python
# Outlier detection 
def detect_outliers(df,n,features):
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # mean
        mean = df[col].mean()
        # standard deviation
        std = df[col].std()
        # the upper bound
        top = mean + std * 1.96
        #  the lower bound 
        bot = mean - std * 1.96
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < bot) | (df[col] > top)].index       
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers 
```

- `Counter`: A [`Counter`](https://docs.python.org/3/library/collections.html#collections.Counter) is a [`dict`](https://docs.python.org/3/library/stdtypes.html#dict) subclass for counting hashable objects. It is an unordered collection where elements are stored as dictionary keys and their counts are stored as dictionary values.

```python
# detect outliers from Age, SibSp , Parch and Fare
outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])
train.loc[outliers_to_drop] # Show the outliers rows

# and remove them
train = train.drop(outliers_to_drop, axis = 0).reset_index(drop=True)
```

### Complementary Functions

Most Machine Learning algorithms cannot work with missing values, so let’s create a few functions to take care of the missing values.

- Check if there're missing values:

  `isnull().any()`

- Calculate how many missing values:

  `isnull().sum()`

We can use the mean + randomized standard deviation to complete the missing data.

Take `Age` as an instance:

```python
# Fill the missing data in Age using mean + randomized standard deviation. 
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)  
    df_age = dataset['Age'].copy()
    df_age[np.isnan(df_age)] = [age_null_random_list]
    dataset['Age'] = df_age.astype(int)
```



## Feature Enginerring

Qualitative data is often nominal (e.g. names) or categorical (e.g. sex). Those can't be ordered and are difficult to evaluate. Therefore we want to convert all our variables to quantitiative data, i.e. numerical or ordinal values.

For example, we can convert the `name` to attribute based on their length

```python
for dataset in full_data:
    try:
        dataset['Name_length'] = train['Name'].apply(len)
    except:
        print("Name_length feature is located in the data frame")
        
train['Name_length'].head()
```

```
0    23
1    51
2    22
3    44
4    24
Name: Name_length, dtype: int64
```

```python
fig, ax = plt.subplots(2,1,figsize=(20,10))

# The amount of survived people by Name length.
sum_Name = train[["Name_length", "Survived"]].groupby(['Name_length'],as_index=False).sum()
sns.barplot(x='Name_length', y='Survived', data=sum_Name, ax = ax[0])
ax[0].set_title('The amount of survived people by Name length')

# The amount of survived people by Name length.
average_Name = train[["Name_length", "Survived"]].groupby(['Name_length'],as_index=False).mean()
sns.barplot(x='Name_length', y='Survived', data=average_Name, ax = ax[1])
ax[1].set_title('The average survival rates')

plt.show()
```

![output_53_1.png](https://github.com/EckoTan0804/HexoBlog/blob/Data-Processing-With-Py/source/images/Exercise_01_Data_Processing_with_Python/output_53_1.png?raw=true)

From the graphics above we can see that all passengers with long names have survived, perhaps the cause is that rich families tend to have longer names.

- [`groupby`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html): Group series using mapper (dict or key function, apply given function to group, return result as series) or by a series of columns.

It can also be helpful to create meaningful "bins" for attributes. 

Therefore we will divide the `Name_length` feature into small classes. Each of these classes has a similar rate to survive.

```python
for dataset in full_data:
    dataset.loc[ dataset['Name_length'] <= 23, 'Name_length']= 0
    dataset.loc[(dataset['Name_length'] > 23) & (dataset['Name_length'] <= 28), 'Name_length']= 1
    dataset.loc[(dataset['Name_length'] > 28) & (dataset['Name_length'] <= 40), 'Name_length']= 2
    dataset.loc[ dataset['Name_length'] > 40, 'Name_length'] = 3
train['Name_length'].value_counts()
```

```
0    360
1    240
2    201
3     90
Name: Name_length, dtype: int64
```

As a next step we can map categorical attributes to a numerical discrete value:

```python
# Mapping Gender
for dataset in full_data:
    # np.where takes as input a list of Booleans, a new value and a backup value
    try:
        dataset['Sex'] = np.where(dataset['Sex']=='female', 1, 0)
    except:
        print('The value is already converted ')
train['Sex'].head()
```

For example we can look at the *Age* attribute:

- [`sns.FacetGrid`](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html): Multi-plot grid for plotting conditional relationships.
- [`sns.kdeplot`](https://seaborn.pydata.org/generated/seaborn.FacetGrid.html): Fit and plot a univariate or bivariate kernel density estimate.

```python
# plot distributions of age of passengers who survived or did not survive
a = sns.FacetGrid( train, hue = 'Survived', aspect=4 )
a.map(sns.kdeplot, 'Age', shade= True )
a.set(xlim=(0 , train['Age'].max()))
a.add_legend()
```

![output_60_1.png](https://github.com/EckoTan0804/HexoBlog/blob/Data-Processing-With-Py/source/images/Exercise_01_Data_Processing_with_Python/output_60_1.png?raw=true)

We can see that until the age of 14 the chance of survival is higher than the chance to die. In reverse the chance for dying is higher between the age of 14 and 30. This changes a couple of times between various ages.

Therefore the best categories for age are:

- 0: less than 14
- 1: 14 to 30
- 2: 30 to 40
- 3: 40 to 50
- 4: 50 to 60
- 5: 60 and more

```python
for dataset in full_data:
    dataset.loc[ dataset['Age'] <= 14, 'Age_bin'] 						             = 0
    dataset.loc[(dataset['Age'] > 14) & (dataset['Age'] <= 30), 'Age_bin'] = 1
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age_bin'] = 2
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age_bin'] = 3
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age_bin'] = 4
    dataset.loc[ dataset['Age'] > 60, 'Age_bin'] 							             = 5
train['Age_bin'].value_counts()
```

The next step is to map the `Embarked` feature:

```python
# Mapping Embarked
for dataset in full_data:
  try:
      dataset.Embarked.replace(('S','C','Q'), (0,1,2), inplace = True)
  except:
      print('The value is already converted ')
train['Embarked'].head()
```

```
0    0.0
1    1.0
2    0.0
3    0.0
4    0.0
Name: Embarked, dtype: float64
```

Additionally data might be skewed. For example, if we look at the `Fare` attribute, we can see it is heavily skewed to the left:

```python
fig, ax = plt.subplots(figsize=(10,5))
sns.distplot(train["Fare"][train["Survived"] == 0], color="r")
sns.distplot(train["Fare"][train["Survived"] == 1], color="b")
```

![output_64_1.png](https://github.com/EckoTan0804/HexoBlog/blob/Data-Processing-With-Py/source/images/Exercise_01_Data_Processing_with_Python/output_64_1.png?raw=true)

**To reduce the skewedness of this attribute, we can transform it with the log function.** This redistributes the data:

![output_66_1.png](https://github.com/EckoTan0804/HexoBlog/blob/Data-Processing-With-Py/source/images/Exercise_01_Data_Processing_with_Python/output_66_1.png?raw=true)

Now we can define bins more easily: The survival rate is lower for a Fare_log value less than 2.7 and higher for values greater than 2.7.

```python
for dataset in full_data:
    dataset.loc[ dataset['Fare_log'] <= 2.7, 'Fare_bin'] = 0
    dataset.loc[ dataset['Fare_log'] > 2.7, 'Fare_bin'] = 1
    dataset['Fare_bin'] = dataset['Fare_bin'].astype(int)
train['Fare_bin'].value_counts()
```

```
0    457
1    434
Name: Fare_bin, dtype: int64
```



## Feature Selection

Feature selection methods can be used to identify and remove unneeded, irrelevant and redundant attributes from data that do not contribute to the accuracy of a predictive model or may in fact decrease the accuracy of the model.

Fewer attributes are desirable because it reduces the complexity of the model, and a simpler model is simpler to understand and explain.

**Which features within the dataset contribute significantly to our goal?**

To calculate the covariance matrix, we should first remove all remaining string attributes:

```python
train.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null int64
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null float64
Name_length    891 non-null int64
Fare_log       891 non-null float64
Fare_bin       891 non-null int64
dtypes: float64(4), int64(8), object(3)
memory usage: 104.5+ KB
```

As you can see, we will drop the features:

- Name
- Ticket
- Cabin

```python
# Feature selection
drop_elements = [ 'Name', 'Ticket', 'Cabin']
try: 
  train = train.drop(drop_elements, axis = 1)
  test  = test.drop(drop_elements, axis = 1)
except:
  print("The features are already removed.")
```

### Correlation Analysis - Multi-variate Analysis

- Basically, correlation measures how closely two variables move in the same direction. Therefore we try to find whether there is a correlation between a feature and a label. In other words as the feature values change does the label change as well, and vice-versa?
- The data may contain a lot of information redundancy distributed among multiple variables, which is a problem called multivariate correllation.

**Heatmap for the correlation matrix**

```python
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
plt.show()
```

[`sns.heatmap()`](https://seaborn.pydata.org/generated/seaborn.heatmap.html): Plot rectangular data as a color-encoded matrix

![output_75_0.png](https://github.com/EckoTan0804/HexoBlog/blob/Data-Processing-With-Py/source/images/Exercise_01_Data_Processing_with_Python/output_75_0.png?raw=true)

We can see from the **survived** column, that it has strong relation with *sex* (0.54) and potential relation with *class* (0.34) (or *fare*).

Therefore, we will drop the features that are not correlated with our dataset.

```python
# Feature selection
drop_elements = ['PassengerId', 'SibSp', 'Age','Embarked']

train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)
```



## Predictive Modelling

```python
import sklearn         # Collection of machine learning algorithms
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
```

### Shuffle and Split Data

The next step requires that we take the train dataset and split the data into training and testing subsets. We should do this because we want to test how well our model generalizes to unseen data.

Use `train_test_split` from `sklearn.cross_validation` to shuffle and split the features and prices data into training and testing sets.

- Split the data into 70% training and 30% testing.
- Set the *random_state* for train_test_split to 101. This ensures results are consistent over multiple runs.

>  [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

```python
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]


from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=101)

```

### Training and Predicting

- Training: `fit()`
- Prediction: `predict()`
- Evaluation: `score()`

####  Logistic Regression

[Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) is machine learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (survived) or 0 (not survived). In other words, the logistic regression model predicts P(Y=1) as a function of X (Features). This makes it a binary classifier.

```python
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred1 = logreg.predict(x_test)
acc_log = round(logreg.score(x_test, y_test) * 100, 2)
acc_log
```

```
77.99
```

#### Decision Tree

Decision tree classifiers are attractive models if we care about interpretability. As the name decision tree suggests, we can think of **this model as breaking down our data by making decisions based on asking a series of questions.**

```python
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred7 = decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_test, y_test) * 100, 2)
acc_decision_tree
```

```
73.88
```

#### Perceptron

The [perceptron](https://en.wikipedia.org/wiki/Perceptron) is a supervised binary classifier.

```python
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred4 = perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_test, y_test) * 100, 2)
acc_perceptron
```

```
56.72
```

### Support Vector Machines

[Support Vector Machines](https://en.wikipedia.org/wiki/Support_vector_machine) (SVM) are kernel based methods that require only a user-specified kernel function 𝐾 i.e., a similarity function over pairs of data points into a kernel (dual) space on which the learning algorithms operate linearly.

```python
svc=SVC()
svc.fit(X_train, Y_train)
Y_pred2 = svc.predict(x_test)
acc_svc = round(svc.score(x_test, y_test) * 100, 2)
acc_svc
```

```
75.37
```



## Conclusion

- Machine Learning is about algorithms that are **capable to learn from data**, instead of having to explicitly code rules.
- In an ML project you **gather data in a training set, and you feed the training set to a learning algorithm.**
- The system will not perform well if your training set is too small, or if the data is not representative, noisy, or polluted with irrelevant features (garbage in, garbage out).
- Lastly, your model needs to be neither too simple (in which case it will underfit) nor too complex (in which case it will overfit).



