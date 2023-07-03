## WHAT IS A PROPENSITY MODEL ?
A statistical technique used to forecast the possibility or probability that an event or behaviour will occur in the future is propensity modelling, sometimes referred to as predictive modelling or propensity score modelling. It entails creating a predictive model based on previous data to calculate the likelihood or propensity of a particular result.
Machine learning algorithms and statistical methods like logistic regression, decision trees, or neural networks are frequently used to create propensity models. These models calculate the probability of the desired event using historical data and a collection of pertinent attributes or predictors. Various client traits, previous behaviours, transactional data, demographic data, and other environmental variables can all be predictors.
In context of the project, Understanding and forecasting client preferences and behaviours depends on propensity modelling.

### In context of this project ->

Banks can improve their marketing strategies and maximise their efforts to attract new customers and keep existing ones by utilising historical data and implementing propensity modelling approaches.
Banks can use propensity modelling to predict a customer's likely or inclination to buy a certain financial instrument, such as a credit card, consumer loan, or mutual fund. Banking institutions can target consumers who are more likely to be interested in certain financial products using the information gained through propensity modelling. Banks may increase conversion rates and maximise their marketing budget by concentrating their efforts and messaging on high-propensity clients.

## APPROACH WORKFLOW 

![PROJECT WORKFLOW](https://github.com/kanha-gupta/Propensity-model-for-Financial-products/assets/92207457/94b04241-8cd4-4eee-b856-7b7ff7d22a75)


## DATA PREPROCESSING

We use the dataset's all sheets to create a new dataset for Testing and Training our model. 
Then, we perform Preprocessing for both consumer loan & mutual funds

-- Firstly we obtain all the original dataset sheets value

-- Then we check each dataset shape. 

-- Then we drop ‘Clients’ due to difference in inflow and outflow dataset.

-- Then we merge all 4 datasets to obtain.

Resolving null errors : 
1.  For Sex, two rows are missing so I impute it with U (unknown) 
2. There are rows missing in Inflow outflow dataset so I impute it with 0
3. I put 0 in every other null values considering client dont avail these features from the bank.
Since dataset is very small, I did not use mean or median as it would lead to smaller variance  & bias to our model.
4. I added 10 years to incorrect ages (Assumption) . We also use knn imputing

## ML model selection process : 

During the selection process, there were following algorithms considered : 

### 1. For Sales probability : 

- Xgbclassifer

- Random forest classifier 

- Logistic regression

- Naive bayes algorithm

- Adaboost classifier

Out of which I chose logistic regression model

### 2. For revenue prediction

- random forest regressor

- ada boost regressor

- ridge

Out of these I chose random forest model

## Project Features


## How it benefits ?

- The project is a demonstation of a type of a propensity model which deals with sales & revenue probability
- For banks & other such financial institutions, such models help them in identifying customers with highest chances of conversion so that they can specially focus on them during marketing & sales so as to maximise their  revenue.
- It also provides a recommendation system which suggest banks as to which product should be advertised/offered to a specific customer so that they dont waste time on suggesting products with low chances of conversion.
- These types of model can be expanded further into more Banking products & services.

## Possible Use Cases

A model of this nature can be applied almost every field where marketing and sales is a concern. This model can also revolutionise Digital marketing because it would reduce the cost significantly after advertisers obtains enough data to train the model.



## **HOW TO SETUP ?**
  
1. Python version 3.10 is used 
2. Enter following command to install packages : `pip install -r requirements.txt`

## ABOUT DATASET 

The dataset is in XLSX file format & contains 4 sheets.
Dataset contains customer actions & buying behaviour of banking tools such as mutual funds, credit card & consumer loan.

dataset used:
[Click here](https://www.kaggle.com/datasets/khushigpt11/financial-dataset-for-propensity-modelling)



   

