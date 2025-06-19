# Titanic Survival Predictor

Building a machine learning model to predict whether a passenger survived the Titanic disaster based on features like age, class, gender, etc.

## Dataset
https://www.kaggle.com/competitions/titanic/data

## Loading & Exploring the Titanic data 

We use the following libraries **pandas, matplotlib, seaborn**. 
To understand the data, visualize the relevant information such as:
- Survival rate (Survived vs Not Survived).
- Survival based on passenger class **(Socioeconomic status)**
- Survival based on age / sex **(Age/Sex bias)** 
- Survival based on the embarking port **(Q, S or C)**

## Cleaning & Preparing Data

First thing, we check for missing values using **df.isnull().sum()**. this shows how many null items are in our data, which we would then handle before starting training.

```python
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```

- For Age, since it is an important feature, we want to use the median for null values since it is robust against outliers
- For Cabin, 77 % of values are null, which is a very large number. We will drop this column entirely.
- For embarked, only two values are missing, for which we will use the mode **(most common)**.

Next thing, we drop the columns which are not relevant for our study (PassengerId, Name, Ticket)
Lastly, we convert the categorical values to numbers:
- Embarking becomes Embarking_Q & Embarking_S, with Embarking_C as the reference.
- Sex becomes Sex_Male (1 for male & 0 for female)

#### Summary 
We generally follow this approach in the data cleaning and preperation phase:
1. Handle Missing Values **= Avoid errors in training**
2. Drop Useless Columns **= Reduce noise & simplify model**
3. Encode Categorical Data **= Convert strings into numbers**
4. Normalize or Scale (if needed) **= Helps some models converge better**

## Train & Evaluate a Classifier

First, we seperate the target y from the inputs, and then we split the data into **training data & test data** through the **train_test_split** function which is found in **sklearn.model_selection**. We then proceed to choose and train a classifier : **Random Forest Classifier** in our case (found in **sklearn.ensemble**)
Finally we can evaluate the accuracy (accuracy score, classification report & confusion matrix) which is found in **sklearn.metrics**
```python
Accuracy Score: 0.8156424581005587
Classification Report:               
                 precision    recall  f1-score   support

           0       0.83      0.86      0.85       105
           1       0.79      0.76      0.77        74

    accuracy                           0.82       179
   macro avg       0.81      0.81      0.81       179
weighted avg       0.81      0.82      0.82       179

Confusion Matrix: [[90 15]
                  [18 56]]
 ```

#### Accuracy Score : 81.5%

#### Classification Report

| Metric        | Means...                                                              |
| ------------- | --------------------------------------------------------------------- |
| **Precision** | Out of the people the model said “survived”, how many really did?     |
| **Recall**    | Out of all people who actually survived, how many did the model find? |
| **F1-score**  | A balance between precision and recall                                |
| **Support**   | Number of actual passengers per class (0 = died, 1 = survived)        |

#### Per Class

| Class | Label    | Notes                                                                             |
| ----- | -------- | --------------------------------------------------------------------------------- |
| **0** | Died     | Precision: 83%, Recall: 86% → Model is **very good at detecting non-survivors**   |
| **1** | Survived | Precision: 79%, Recall: 76% → A bit weaker at detecting survivors, but still good |

The imbalance is expected since more people died than survived. The model naturally learns to favor the majority class.

#### Confusion Matrix

```python
[[90 15] # 90 predicted died (true positive) & 15 predicted survived (false negative) 
 [18 56]] # 18 predicted died (false negative) & 56 predicted survived (true negative)
```

**Interpretation :** 
- 146 predictions were correct (90 & 56).
- 33 were wrong (18 & 15).

## 2nd Classifier = Logistic Regression

Using the same method , we try another classifier : **Logistic Regression** found in **sklearn.linear_models**.
We get the following results:
```python
Accuracy Score: 0.8100558659217877
Classification Report:               
                 precision    recall  f1-score   support

           0       0.83      0.86      0.84       105
           1       0.79      0.74      0.76        74

    accuracy                           0.81       179
   macro avg       0.81      0.80      0.80       179
weighted avg       0.81      0.81      0.81       179

Confusion Matrix: [[90 15]
                  [19 55]]
 ```

#### Accuracy Score : 81%

Logistic Regression is **nearly as good** as Random Forest. Random Forest might edge ahead with tuned hyperparameters or more features.

## Making Predictions on actual data 

Just like the training data , we do the same preprocessing with real data, and then make the prediction. We then register the results in submission.csv .
And just like that, we complete a full ML pipeline

| Phase               | Content                                                    |
| ------------------- | ---------------------------------------------------------- |
| **Data Cleaning**   | Handled missing values, dropped noise, encoded categories  |
| **EDA**             | Explored survival patterns with logical and visual tools   |
| **Model Training**  | Trained **Random Forest** and **Logistic Regression**      |
| **Evaluation**      | Used accuracy, precision, recall, F1, and confusion matrix |
| **Real Prediction** | Preprocessed unseen test data and made predictions         |

## Conclusion

Built and evaluated two models (Random Forest, Logistic Regression) to predict survival on the Titanic dataset.  
Performed data cleaning, encoding, and model training using scikit-learn.  
Achieved ~81% internal accuracy; Kaggle submission scored **0.74401**.  
This project covers the full ML workflow from preprocessing to prediction.
