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

