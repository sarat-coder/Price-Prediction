# Price-Prediction
A Complete ML on Regression
__________________________________________
# Problem Statement and Wayforward
### 1. Predicting the district's median housing price, which will be fed into another ML system along with many other signals.
### 2. The predictions are currently estimated manually by experts by using complex rules.
### 3. This manual process is costly and time consuming and the estimates are not great.
### 4. Hence management suggested to go ahead with ML approach.
____________________________________________
1. Loading and reading the dataset.
2. Understanding attributes.
Undestanding attributes
Each row represents one district, There are 20640 districts.
Each column represents one feature, there are 10 features.
total_bedrooms, is having lesser number of records (207) we have found from df.info() command. We have to handle these missing values.
All attributes are float expect ocean_proximity , which is Object and a categorical field.Value_counts function explains (1H Ocean having 9136 records,Inland is 6551,near ocean is having 2658 districts, Near bay having 2290 districts and island 5 districts.
Describe function explains:
  1. std: standard deviation i.e. dispersion from mean.,
  2. 25% (first quartile), 50%(2nd quartile(median)),75%(third quartile) represents the percentile: (e.g.housing_median_age, 25% districts have a housing_median_age less than 18, 50% districts are lower than 29 and 75% are lower than 37)
  3. Count: explains how many records we are havinv on that attribute.
  4. mean: average of observation in that attribute.
  5. min: minimum of observation of the attribute.
  6. max: maximum of observation of the attribute.
  
------------------------------------------------
# hist() explains:
    * median_income is not in USD, it is actually tens of thousands of dollars. e.g 3 mean $ 30,000 dollars.
    * median_income is capped at 15 for higher incomes and 0.5 for lower income.
    * Similarly housing_median_age and median_house_value both are capped
    * As our target columnm is median_house_value, which our machine is going to predict, may lean that price never go beyond 50,00000 higher capped value.
    * If we are going to build a ML which need to predict beyond even capping value there there are two options before processing the data.
        a) Collecting proper labels for districts which labels are capped.
        b) Remove those districts from training and test set 
    * We need to do feature_scaling, as all these attributes are in different scales.
    * Most of histograms are right_tailed, so we need to normalize (bell-shaped) the data as this may be a big problem for some ML models to detect the patterns (relationships).
 ------------------------------------------------
 
 # hist() explains:
    * median_income is not in USD, it is actually tens of thousands of dollars. e.g 3 mean $ 30,000 dollars.
    * median_income is capped at 15 for higher incomes and 0.5 for lower income.
    * Similarly housing_median_age and median_house_value both are capped
    * As our target columnm is median_house_value, which our machine is going to predict, may lean that price never go beyond 50,00000 higher capped value.
    * If we are going to build a ML which need to predict beyond even capping value there there are two options before processing the data.
        a) Collecting proper labels for districts which labels are capped.
        b) Remove those districts from training and test set 
    * We need to do feature_scaling, as all these attributes are in different scales.
    * Most of histograms are right_tailed, so we need to normalize (bell-shaped) the data as this may be a big problem for some ML models to detect the patterns (relationships).
1
# hist() explains:
2
    * median_income is not in USD, it is actually tens of thousands of dollars. e.g 3 mean $ 30,000 dollars.
3
    * median_income is capped at 15 for higher incomes and 0.5 for lower income.
4
    * Similarly housing_median_age and median_house_value both are capped
5
    * As our target columnm is median_house_value, which our machine is going to predict, may lean that price never go beyond 50,00000 higher capped value.
6
    * If we are going to build a ML which need to predict beyond even capping value there there are two options before processing the data.
7
        a) Collecting proper labels for districts which labels are capped.
8
        b) Remove those districts from training and test set 
9
    * We need to do feature_scaling, as all these attributes are in different scales.
10
    * Most of histograms are right_tailed, so we need to normalize (bell-shaped) the data as this may be a big problem for some ML models to detect the patterns (relationships).
------------------------------------------------

Sampling and creating training and test sets
There are two type of sampling
1. Random Sampling
2. Stratified Sampling
Random Sampling
This works fine if the dataset is large enough ralative to the number of attributes. In this case it may be skewed in test set sampling , results would be significantly biased.
from sklearn.model_selection import train_test_split train_set,test_set=train_test_split(housing,test_size=.2,random_state=42)

Stratified Sampling e.g if total population is 1000,and there are 51% belongs to male and 49% belongs to female, then the ration of sampling should be apprrox 5100 for male adn 4900 for female. The whole population is divided into homogenious subgroups called strata. This helps to guarantee the test set represents the overall population.
Should not have too many strata and each stratum should be large enough to avoid from baising.
-----------------------------------
![image](https://user-images.githubusercontent.com/65825617/124791317-93d98b80-df69-11eb-8876-b30c2364a4c5.png)
### Let us visualize the polulation vs price
    * price represents the colors, using cmap =jet colour map
    * Blue represents low values
    * Red represents high values.
    * Circle represents Population
![image](https://user-images.githubusercontent.com/65825617/124791583-cedbbf00-df69-11eb-847a-27bd3793a16f.png)

### This image represents the Prices are very much related to the density and the location (close to ocean)

### The most promising attribute to predict the median house value is the median income. Let's zoom and analyze
![image](https://user-images.githubusercontent.com/65825617/124791888-1b26ff00-df6a-11eb-99d9-6d66edc13aa2.png)

### Correlation indeed very strong.
    * The price cap cleary visible aa a horizontal line at $5000000, another at $450000, and at $350000
    * Hence few data quirks are there.
    * Some attributes are heavy-tailed distribution.
  
### Rather to understand numbe of rooms in a district, better to know how many households are there (rooms per household)
### similarly rather to understand the no of bedrooms, better to compare with no of rooms.(bedoroms per room)
### Similary population per household.(population per household)
### So let's create new attributes

![image](https://user-images.githubusercontent.com/65825617/124792084-4ad60700-df6a-11eb-840d-e6637ac651e6.png)

### Rather than total_rooms, the ration bedrooms_per_room is highely correlated with the house price.
### That means houses having lesser bedrooms/room ratio tends to more expensive.
### rooms_per_household is more correlated than total_rooms in the district that means larger houses, the more they expensive.


### 1. Data Cleaning
* 1.1 (Handling Missing Values)
    * There are three ways
        * get rid of corresponding rows which have null values.
        * get rid of whole attribute that contains null values.
        * set the values to some value (zero,mean,median, etc..)
                * by using fillna
                * by using sklearn SimppleImputer
* 1.2 (Handling Text and Categorical Attributes)
        * OrdinalEncoder
        * OneHotEncoder
* 1.3 Custom Transformers
* 1.4 Feature Scaling.

# As we know "total_bedrooms" is having null values.
# option-1: get rid of corresponding rows which have null values.
# housing_train.dropna(subset=["total_bedrooms"])
# option-2: get rid of whole attribute that contains null values
# housing_train.drop("total_bedrooms",axis=1)
# option-3.a: set the values to some value (zero,mean,median, etc..) fillna
# median=housing_train["total_bedrooms"].median()
# housing_train["total_bedrooms"].fillna(median,inplace=True)
# option-3.b: set the values to some value (zero,mean,median, etc..) Sklearn SimpleImputer

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy="median")

# please note Simpleimputer can be applied to onlyl numerical attributes. so we need to copy of the data without text features

housing_num=housing_train.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)

print(imputer.statistics_)
x=imputer.transform(housing_num) # it is an array

# converting array to dataframe
housing_tr=pd.DataFrame(x,columns=housing_num.columns,index=housing_num.index)

# Handling Text and Categorical Attributes
# As most of ML models perform to work witht numbers.So let us convert this categorical from text to number.
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder=OrdinalEncoder()
housing_cat_encoded=ordinal_encoder.fit_transform(housing_cat)

### OrdinalEncoder, represents in numerical form starting from 0 to 4 (five categories)
### Issue with this representation, it seems nearby values are more similar that distance values.
### Hence this technique is good for ordinal categorical features like ("bad","good","average","excellence")
## To fix this issue we will use one-hot encoding by creating dummy binary attributes

# Customer Transformers
    * What we have learn to transform the attributes, let us create a custom function
    
![image](https://user-images.githubusercontent.com/65825617/124792560-b91ac980-df6a-11eb-9c43-10ccf13aa53a.png)

### ML algorithms not perform well if the inpput numerical attributes are not in one scale or they are in different scales.
### In our dataset we have observed that , total number of romms ranges from 6 to 39k while income ranges from 0 to 15
### Scaling of target value is not required.
-----------------------
# Feature Scalling
#### There are two type of scalling:
                * a. MinMaxScaler (rescaled all values and normalized between o and 1)
                * b. standardization
    . MinMaxScaler: subtracting the min value and dividing by the max minus min.( give value between o and 1)
    . Standardization: first it subtracts the mean value and then divies by standard deviation so the distribution 
    has unit variance and will always have zero mean.
---------------------
# Transformation PIPELINE
### So we have handled numerical attributes and categorical attributes seprately.
### it is better to have single transformer which can able to handle both, for this puprpose we can use ColumnTransformer of SKLEARN.
### LET US CREATE A PIPELINE FOR NUM ATTRIBUTES TO TRANSFER THE FOLLWOING ACTIVITES.
    * SIMPLEIMPUTER
    * COMBINEATTRIBUTEADDER (CUSTOMIZED ONE)
    * STANDARD SCALLER.
### THEN WE WILL USE COLUMNTRANSFORMER TO HANDLE BOTH NUM AND CATEGORICAL FIELDS (ONEHOT ENCODING)

![image](https://user-images.githubusercontent.com/65825617/124797675-27ae5600-df70-11eb-9bb9-d038db31a726.png)

-----------------------------------

# Selecting and Training the Model
-----------------------------
1. Linear Regressiong.
The result shows.
### Linear Regressor is not a good prediciton in training set that means model is underfitting.
### This indicates that the features are not sufficient for good predictions and the model we have used is not a powerful one
--------------------------
2. Decision Tree.
### In Decision Tree is n, there is no error.
### This indicates that it's badly overfitted.
# Let us use Cross validation to evaluate both models.
    * if we define cv=10, then it randomly splits the training set into 10 distint folds/subsets.
    * then it trains and evalutes 10 times.
    * by picking a different fold for validation and 9 other folds for training
    * it will give 10 evaluation scores
 # Findings
### The Decision Tree is overfitting so badly that it performs worse than Linear Regression
-----------------------------
3. Random Forest.
# Let us try another model the last one -RandomForestRegression
### It has been observed that with CV, Random Forest peforms very well.
### However noticed that the train score 18718 is much lesser than the validation score 50246.
### This means the model is still overfitting.
### So we need to simplify the model by using regularization technique.
------------------------
# Fine-Tune the MODEL
### in first evaluation it will run for 3*4=12 combinations of hyper parameters
### in second evaluation it will run for 2*3=6 combinations but this tyme bootstrap= False (bydefault it was true)
### So in total grid_search model will run with a combination of 12+6=18 and it will train for 5 times (cv=5)
### so total there will be 18*5=90 rounds of training.

### THE BEST RESULT IS 49935.20703733164 {'max_features': 8, 'n_estimators': 30
### our RF model is best AT max_features=8 and n_estimators=30, with rmse result 49935

---------------------
# Analysing the best model and their errors
#### Finding feature importancees (grid_search.best_estimator_.feature_importances_)

### In training the error is 49935 where as in testing it is 48404
### To know  how precise this estimate is let us compute at 95% confidence interval for the generalilzation of model

---------------------
Conclusion...
![image](https://user-images.githubusercontent.com/65825617/124797192-a060e280-df6f-11eb-91cd-b7e9efdbf9fb.png)


