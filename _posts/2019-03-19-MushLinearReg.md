---
title: "Machine Learning Projects: Logistic Regression"
date: 2019-03-19
tags: [Machine Learning]
header:
  image: "/images/mushrooms1.jpg"
excerpt: "Data Science Projects, Machine Learning, Classification"
mathjax: "true"
classes: wide
---
# Mushroom Classification With Logistic Regression

## Aim

I'm working with a dataset that contains 8124 different mushrooms. My ultimate goal is to determine whether a mushroom is edible or poisonous. I will achieve this is by using the different features of each mushroom and find a correlation between the features and the class (poisonous/edible). There are some obvious benefits of this classification, the first benefit is safety. Being able to classify a poisonous mushroom using a few features, such as the cap shape, cap color, odor, and gill size for example could save peoples lives. The alternative would be to try to determine the chemical composition of the mushroom or to have access to the entire list of poisonous mushrooms. Both of these options would be extremely impractical for hikers or people living in remote areas. Hence being able to classify a mushroom accurately using its features would be highly beneficial.

## Strategy

I'm going to perform this classification using the machine learning technique, logistic regression. I've decided to use logistic regression because it's one of the simplest classification methods. This gives me the ability to clearly explain each step in this project, without the project getting too long. The aim of logistic regression is to separate the data using a linear function, this function is called the decision boundary. All the points on one side of the decision boundary are classified as poisonous and all points on the opposite side are classified as edible. I'm not going to explain explicitly how the linear function is determined, because this will require university level mathematics and statistics. My short explanation is that each muchroom has a probability function assigned to it, which determines the probability of achieving the correct class (poisonous/edible), given its features. Then the probability functions for all mushrooms are combined to form another function, called a loss function. Then I need to find the minimum of this loss function, which will tell me the optimal decision boundary. I'm going to use skikit-learn to do the logistic regression, so you won't see any of these calculations in this project. Logistic regression can be performed on data points with many dimensions, but for this project I want to work with data in two dimensions. This will allow my computer to quickly process the data and give me the ability to plot all the points on a two dimensional graph. I will end up having 95 separate dimensions, so to transform these features into 2 dimensions, I will use a technique called principle component analysis (PCA). PCA uses a fairly advanced mathematical technique called projection, which projects a vector with higher dimensions onto a vector with lower dimensions. I will be relying on skikit-learn to perform PCA, so you won't need to know exactly how it works. Once the features have been reduced to two dimensions, then I will plot the points on a graph and add the linear decision boundary, created from the logistic regression classifier.

## Importing Libraries and Loading Data

The first thing I need to do is import all the tools I'll need, such as numpy for linear algebra operations, pandas for creating/manipulating dataframes and matplotlib for creating graphs.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
```


```python
data = pd.read_csv('mushrooms.csv')
```

## Analysing The Data

Observing the data below, you can see that each row represents a separate mushroom. The first column, 'class', has two options, p and e, which represent poisonous and edible respectively. Hence this is the column I'm trying to predict accurately. The rest of the columns are the features I will be using to accurately predict the class.


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>p</td>
      <td>x</td>
      <td>s</td>
      <td>n</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>1</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>y</td>
      <td>t</td>
      <td>a</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>g</td>
    </tr>
    <tr>
      <th>2</th>
      <td>e</td>
      <td>b</td>
      <td>s</td>
      <td>w</td>
      <td>t</td>
      <td>l</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>n</td>
      <td>n</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>p</td>
      <td>x</td>
      <td>y</td>
      <td>w</td>
      <td>t</td>
      <td>p</td>
      <td>f</td>
      <td>c</td>
      <td>n</td>
      <td>n</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>k</td>
      <td>s</td>
      <td>u</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>x</td>
      <td>s</td>
      <td>g</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>w</td>
      <td>b</td>
      <td>k</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>e</td>
      <td>n</td>
      <td>a</td>
      <td>g</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



I need to make sure the data is clean before I can proceed. As you can see below, there are no null values. This is a very good sign. I will have to assume there is no mislabeled data, because there is simply no way for me to check this.


```python
data.isnull().sum()
```




    class                       0
    cap-shape                   0
    cap-surface                 0
    cap-color                   0
    bruises                     0
    odor                        0
    gill-attachment             0
    gill-spacing                0
    gill-size                   0
    gill-color                  0
    stalk-shape                 0
    stalk-root                  0
    stalk-surface-above-ring    0
    stalk-surface-below-ring    0
    stalk-color-above-ring      0
    stalk-color-below-ring      0
    veil-type                   0
    veil-color                  0
    ring-number                 0
    ring-type                   0
    spore-print-color           0
    population                  0
    habitat                     0
    dtype: int64



If you look at the amount of unique elements in each column, it ranges from 2 to 12. This will become a problem when I do PCA, because I need each element in a column to be comparable to each other on a scale. For example, you can't measure objectively how close one cap shape is to a different cap shape, or how close an odor is to a different odor on a numerical scale. The solution to this problem is to create a new column for each unique element in a column. The next issue is that the elements in each column aren't comparable to elements in other columns. For example, you can't compare a specific cap shape to an odor. To solve this I will use a skikit-learn function called StandardScalar, which gives each column the same mean = 0, and standard deviation = 1. This essentially makes the elements in each column directly comparable.


```python
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>...</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
      <td>8124</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>2</td>
      <td>6</td>
      <td>4</td>
      <td>10</td>
      <td>2</td>
      <td>9</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>12</td>
      <td>...</td>
      <td>4</td>
      <td>9</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>9</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>top</th>
      <td>e</td>
      <td>x</td>
      <td>y</td>
      <td>n</td>
      <td>f</td>
      <td>n</td>
      <td>f</td>
      <td>c</td>
      <td>b</td>
      <td>b</td>
      <td>...</td>
      <td>s</td>
      <td>w</td>
      <td>w</td>
      <td>p</td>
      <td>w</td>
      <td>o</td>
      <td>p</td>
      <td>w</td>
      <td>v</td>
      <td>d</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>4208</td>
      <td>3656</td>
      <td>3244</td>
      <td>2284</td>
      <td>4748</td>
      <td>3528</td>
      <td>7914</td>
      <td>6812</td>
      <td>5612</td>
      <td>1728</td>
      <td>...</td>
      <td>4936</td>
      <td>4464</td>
      <td>4384</td>
      <td>8124</td>
      <td>7924</td>
      <td>7488</td>
      <td>3968</td>
      <td>2388</td>
      <td>4040</td>
      <td>3148</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 23 columns</p>
</div>



## Transforming The Data

Now I'm separating the data into labels (what I'm trying to predict) and features (what I'm using to predict the labels).


```python
features = data.drop('class', axis=1)
labels = data['class']
```

As I stated previously, each column needs to be numerically comparable. So the first thing I need to do is convert all the string elements to integers. Skikit-learn has a built in function called LabelEncoder, which converts each unique string element into a new integer.


```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in features.columns:
    features[col] = le.fit_transform(features[col])

labels = le.fit_transform(labels)
```

As you can see below, all the string elements have now been converted to integers.


```python
features.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cap-shape</th>
      <th>cap-surface</th>
      <th>cap-color</th>
      <th>bruises</th>
      <th>odor</th>
      <th>gill-attachment</th>
      <th>gill-spacing</th>
      <th>gill-size</th>
      <th>gill-color</th>
      <th>stalk-shape</th>
      <th>...</th>
      <th>stalk-surface-below-ring</th>
      <th>stalk-color-above-ring</th>
      <th>stalk-color-below-ring</th>
      <th>veil-type</th>
      <th>veil-color</th>
      <th>ring-number</th>
      <th>ring-type</th>
      <th>spore-print-color</th>
      <th>population</th>
      <th>habitat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>8</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 22 columns</p>
</div>




```python
labels
```




    array([1, 0, 0, ..., 0, 1, 0])



Most of the integers within each column are not numerically comparable, so I'm going to create a new column for each unique integer.


```python
features = pd.get_dummies(features, columns=features.columns, drop_first=True)
```


```python
features.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cap-shape_1</th>
      <th>cap-shape_2</th>
      <th>cap-shape_3</th>
      <th>cap-shape_4</th>
      <th>cap-shape_5</th>
      <th>cap-surface_1</th>
      <th>cap-surface_2</th>
      <th>cap-surface_3</th>
      <th>cap-color_1</th>
      <th>cap-color_2</th>
      <th>...</th>
      <th>population_2</th>
      <th>population_3</th>
      <th>population_4</th>
      <th>population_5</th>
      <th>habitat_1</th>
      <th>habitat_2</th>
      <th>habitat_3</th>
      <th>habitat_4</th>
      <th>habitat_5</th>
      <th>habitat_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 95 columns</p>
</div>



## Splitting Into Test And Training Data

When creating a logistic regression model, I need to know the class (poisonous/edible) associated with each mushroom. However, I can only understand the true accuracy of a classifier by testing on mushroom data, where the class hasn't already been provided. So I have allocated 70% of my data to creating my logistic regression model and 30% of my data to testing the accuracy of my model.


```python
from sklearn.model_selection import train_test_split
feat_train, feat_test, label_train, label_test = train_test_split(features, labels, test_size = 0.3, random_state=42)
```


```python
feat_train.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cap-shape_1</th>
      <th>cap-shape_2</th>
      <th>cap-shape_3</th>
      <th>cap-shape_4</th>
      <th>cap-shape_5</th>
      <th>cap-surface_1</th>
      <th>cap-surface_2</th>
      <th>cap-surface_3</th>
      <th>cap-color_1</th>
      <th>cap-color_2</th>
      <th>...</th>
      <th>population_2</th>
      <th>population_3</th>
      <th>population_4</th>
      <th>population_5</th>
      <th>habitat_1</th>
      <th>habitat_2</th>
      <th>habitat_3</th>
      <th>habitat_4</th>
      <th>habitat_5</th>
      <th>habitat_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5921</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1073</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3710</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 95 columns</p>
</div>



## Scaling The Data

I now have all my column elements in numeric form and all the elements within a column are numerically comparable to other elements within the same column. However, the elements in each column are still not comparable to elements in other columns. To achieve this, I'm going to scale the data using skikit-learn's StandardScalar function.


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

feat_train = sc.fit_transform(feat_train)
feat_test = sc.transform(feat_test)
```


```python
feat_train
```




    array([[-0.02297586, -0.79088576, -0.34729656, ..., -0.40198689,
             4.72263472, -0.161767  ],
           [-0.02297586, -0.79088576, -0.34729656, ..., -0.40198689,
            -0.21174621, -0.161767  ],
           [-0.02297586, -0.79088576, -0.34729656, ..., -0.40198689,
            -0.21174621, -0.161767  ],
           ...,
           [-0.02297586,  1.26440511, -0.34729656, ...,  2.48764329,
            -0.21174621, -0.161767  ],
           [-0.02297586, -0.79088576,  2.87938356, ...,  2.48764329,
            -0.21174621, -0.161767  ],
           [-0.02297586, -0.79088576,  2.87938356, ..., -0.40198689,
            -0.21174621, -0.161767  ]])



As you can see below the standard deviation of each column is 1, and the means of all the columns are so small, they're essentially 0.


```python
feat_train.std(axis=0)
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])




```python
feat_train.mean(axis=0)
```




    array([-9.99708387e-18, -5.32657125e-17,  2.06189855e-17,  7.49781290e-18,
           -1.74948968e-17,  0.00000000e+00, -6.34190008e-17,  1.74948968e-17,
           -7.49781290e-18,  4.40496508e-17,  4.37372419e-18,  3.74890645e-18,
            3.49897935e-17,  1.74948968e-17,  3.49897935e-17,  5.40467346e-17,
           -6.09197298e-17, -6.99795871e-17,  1.87445322e-17, -4.87357838e-17,
            1.02470110e-16, -9.37226612e-18, -1.12467193e-17, -3.46773847e-17,
           -1.43708081e-17,  4.99854193e-18, -9.80963854e-17, -1.87445322e-17,
            2.74919806e-17,  4.24876064e-17, -4.99854193e-18, -7.31036758e-17,
           -4.99854193e-18,  2.24934387e-17,  7.18540403e-18,  4.99854193e-17,
           -1.68700790e-17, -1.24963548e-17, -2.68671629e-17,  1.24963548e-18,
           -2.43678919e-17, -1.87445322e-18,  2.74919806e-17,  3.12408871e-19,
           -3.12408871e-17, -7.12292225e-17,  1.01220474e-16,  1.12467193e-17,
            1.56204435e-16,  5.99825032e-17,  6.24817742e-18, -9.37226612e-18,
            2.18686210e-17,  7.24788580e-17, -1.37459903e-17,  4.18627887e-17,
           -1.74948968e-17,  5.24846903e-17,  4.37372419e-18, -9.37226612e-18,
           -8.12263064e-18, -3.62394290e-17,  1.62452613e-17,  4.18627887e-17,
           -7.49781290e-18,  2.37430742e-17,  1.49956258e-17,  3.12408871e-18,
            1.39178152e-16,  4.37372419e-18, -9.74715677e-17, -5.12350548e-17,
           -7.49781290e-18,  7.49781290e-18, -9.37226612e-18, -5.49839613e-17,
           -5.99825032e-17, -7.56029467e-17,  5.87328677e-17,  0.00000000e+00,
            2.18686210e-18,  3.68642468e-17, -4.06131532e-17, -3.74890645e-18,
            5.12350548e-17, -8.74744838e-18,  3.90511089e-17,  2.74919806e-17,
            5.74832322e-17, -5.74832322e-17,  5.37343258e-17,  1.24963548e-17,
           -1.56204435e-17, -2.99912516e-17, -5.93576855e-17])



## Principle Component Analysis

Now that all the columns are directly comparable, I can start PCA. I'm going to project from 95 dimensions to 2 dimensions. I will lose some accuracy by projecting onto two dimensions, however having two dimensional data is essential for plotting all the data on a single graph. The purpose of this project is mainly to demonstrate the different machine learning techniques, so I don't have a problem sacrificing some accuracy for this purpose.


```python
from sklearn.decomposition import PCA

feat_train = PCA(n_components=2).fit_transform(feat_train)
feat_test = PCA(n_components=2).fit_transform(feat_test)
```

As you can see below, the features have now been condensed to two dimensions, while most of the information has been retained.


```python
feat_train
```




    array([[ 0.29838967, -1.27843064],
           [-1.22701275, -1.73953456],
           [ 5.31530952, -0.7348945 ],
           ...,
           [-1.73085195, -1.10394368],
           [ 1.53745907,  1.97937363],
           [-1.13663907, -0.14860157]])



## Calculating Accuracy

I'm using skikit-learn's accuracy_score function to measure the accuracy of the classifier.


```python
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score
```

I've created a small function that will determine the accuracy of the classifier on the training data or the test data. You should expect the training data to be more accurate than the test data, because the classifier was built with access to the training class (poisonous/edible).


```python
def print_accuracy(classifier, feat_train, label_train, feat_test, label_test, dataset):
    if (dataset == "train"):
        print("Training Accuracy: {0:.4f} \n" .format(accuracy_score(label_train, classifier.predict(feat_train))))

    elif (dataset == "test"):
        print("Test Accuracy: {0:.4f} \n" .format(accuracy_score(label_test, classifier.predict(feat_test))))        

```

## Creating The Logistic Regression Model

The data is now ready for logistic regression. I just need to feed my training features and training classes into skikit-learns LogisticRegression model, and all the mathematical calculations will be automatically executed for me.


```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

classifier.fit(feat_train, label_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



## Testing The Accuracy

The logistic regression classifier appears to have performed fairly well, with accuracy on the training set at 90.57% and accuracy on the test set slightly lower at 90.11%. These results are very positive, however, it would be beneficial to see where the model is falling short.


```python
print_accuracy(classifier, feat_train, label_train, feat_test, label_test, "train")
```

    Training Accuracy: 0.9057




```python
print_accuracy(classifier, feat_train, label_train, feat_test, label_test, "test")
```

    Test Accuracy: 0.9011



## Plotting The Data

I've created a function to display the data (show_plot), this will provide some useful insights. The function can either display the training or the test data. I achieved this by assigning train/test data to new variables, the x_set and y_set. Then I created a mesh grid, this is a dense rectangle of coordinates, covering the area of the data points. For this plot, I'm using the mesh grid to find the data points on the boundary. However, the mesh grid will become more useful when creating the decision boundary in the next plot. To show all the points, I've created a for loop, with a scatter plot. Every point that has the class 0 (edible) is assigned the color green and every point that has the class 1 (poisonous) is assigned the color red.


```python
def show_plot(dataset):

    plt.figure(figsize = (12,8))

    if (dataset == 'train'):
        x_set, y_set = feat_train, label_train

    elif (dataset == 'test'):
        x_set, y_set = feat_test, label_test


    x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))

    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())

    for i, j in enumerate(np.unique(label_train)):        
        plt.scatter(x_set[y_set == j,0], x_set[y_set == j,1], marker='o',
                    color = ListedColormap(('green', 'red'))(i), label=j)


    plt.xlabel('Princical Component 1', fontsize = 16)
    plt.ylabel('Principle Component 2', fontsize = 16)
    plt.legend()

    if (dataset == 'train'):
        plt.title("Training Set", fontsize = 16)

    elif (dataset == 'test'):
        plt.title("Test Set", fontsize = 16)

    plt.show()
```

The training set shows a significant portion of red (poisonous) points on the right and a significant portion of green (edible) points on the left. However, there is also a mixture of points in the bottom left cluster. This area is where most of the accuracy will be lost. Realistically 100% accuracy will never be achievable for any classifier, because some of the red and green points occupy the same coordinates.


```python
show_plot('train')
```


![](/images/output_59_0.png?raw=true)


The test set has a very similar pattern to the training set, with the red points on the right and green points on the left. The major difference between the sets is that the test set is more sparse, which is exactly what we would expect, since the test set uses 30% of all data points.


```python
show_plot('test')
```


![](/images/output_61_0.png?raw=true)


## Plotting The Decision Boundary

This plot is very similar to the previous plot, except for a few small changes to include the decision boundary. I plotted the decision boundary with the contourf function.


```python
def classifier_plot(model, dataset):

    plt.figure(figsize = (12,8))

    if (dataset == 'train'):
        x_set, y_set = feat_train, label_train

    elif (dataset == 'test'):
        x_set, y_set = feat_test, label_test

    x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))

    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())

    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                 alpha = 0.6, cmap = ListedColormap(('green', 'red')))        

    for i, j in enumerate(np.unique(label_train)):        
        plt.scatter(x_set[y_set == j,0], x_set[y_set == j,1], marker='o',
                    color = ListedColormap(('green', 'red'))(i), label=j)


    plt.xlabel('Princical Component 1', fontsize = 16)
    plt.ylabel('Principle Component 2', fontsize = 16)
    plt.legend()

    if (dataset == 'train'):
        plt.title("%s Training Set" %(model), fontsize = 16)
        print_accuracy(classifier, feat_train, label_train, feat_test, label_test, "train")

    elif (dataset == 'test'):
        plt.title("%s Test Set" %(model), fontsize = 16)
        print_accuracy(classifier, feat_train, label_train, feat_test, label_test, "test")

    plt.show()
```

The decision boundary has separated the red points on the right and the green points on the left very well. As I expected, the bottom left cluster is where all the incorrectly classified points are located. A more complex decision boundary is needed to classify the bottom left cluster more accurately. I will look at a more complex classifier called, kernel support vector machines, in my next project. Then I can compare the two classifier techniques.


```python
classifier_plot('Linear Regression', 'train')
```

    Training Accuracy: 0.9057




![](/images/output_66_1.png?raw=true)


The accuracy of the test set is only 0.46% lower than the training set. As you can see, the misclassified points in the test set are in a very similar position to the misclassified points in the training set.


```python
classifier_plot('Linear Regression', 'test')
```

    Test Accuracy: 0.9011




![](/images/output_68_1.png?raw=true)


## Conclusion

The logistic regression classifier performed fairly well, with 90.11% accuracy on the test set. However, the classifier completely failed to separate the green and red points occupying the bottom left cluster. The last 10% accuracy depends entirely on how well the classifier can separate the bottom left cluster. Logistic regression is one of the simplest classifiers, so achieving such high accuracy with a linear decision boundary is very impressive. I am overall very satisfied with the result of the logistic regression classifier and this leaves me optimistic for my next project involving the kernel support vector machine classifier.
