# Zillow Data Science Exercise

In this exercise sales prices of houses sold in King County, WA in 2015 have been predicted. The prediction model has been trained using the dataset provided in the Data Science ZExercise_TRAINING_CONFIDENTIAL1.CSV and predictions have been made using the data in file Data Science ZExercise_TESTING_CONFIDENTIAL2.CSV

## Data 

The prediction model was trained on 11588 data points, with the validation set for the model being 2318 data points. Columns in the input and prediction dataset which have been altered from their original form are as follows:

**Features**
1.  `TransDate`: The sale date is converted to number of days since the start of the year 2015.
2. `censusblockgroup`: This field has been ommitted from the dataset.
3. `ZoneCodeCounty`: This field is composed of 2 components, an initial code followed by a whitespace and then a key. The key has been ommitted in the prediction analysis and the initial code has been preserved. HashEncoding has been performed to encode this categorical code. 
4. `Usecode`: This field has been ommitted from the dataset.
5. `GarageSquareFeet`: The NA cases in this attribute have been filled as 0. Having no garage is equivalent to 0 square feet of garage and hence the filling. 
6.  `ViewType`: The NA cases in this attribute have been filled as a dummy categorical class. Having no view is equivalent to a class representing the base case.
7. `BGMedHomeValue`: Nearest Neighbors has been used to find values for the missing values. The training data excluded 'BGMedYearBuilt' and 'BGMedRent', which contain some values. 20 nearest neighbors were uniformly weighted and the Euclidean distance metric was used. 
8. `BGMedYearBuilt`: Nearest Neighbors has been used to find values for the missing values. The training data excluded 'BGMedHomeValue' and 'BGMedRent', which contain null values. 20 nearest neighbors were uniformly weighted and the Euclidean distance metric was used.
9. `BGMedRent`: A secondary regression problem to deal with null values for this attribute. The remaning dataset was used to predict values. A RandomForest regressor with mean absolute error criterion, 10 estimators, minimum splitting size as 70 data points and minimum leaf size as 40 data points is used as a regressor. 

**Data Assumptions**
1.  `TransDate`: After the initial transformation, this variable is evaluated on an ordinal scale. Sale price is assumed as a continuous function of the time of the year. Any seasonality in the price changes would be captured by the model under this assumption.

2. `ZoneCodeCounty`: The second component of this field has been ommitted for this analysis. This reduces state space size. It helps the model to generalise to data points where the we encounter an unseen second component. The validity of this method relies on the assumption that the second component is a sub class within the first code. The retained part is coded as a categorical variable. 

3. `View Type`: This attribute is treated like an ordinal attribute in the prediction process. The prediction model is non-linear and hence learns a mapping for each view type to the house sale price, assuming that we have seen the different types of view types in our training set. 

The resulting processing steps result in a training set of 11588 data points and 27 attributes. All attributes in the training set are centered around their mean and their variance is scaled to 1. 


## Model construction, training and validation 

The resultant training dataset is split into a training set and a validation set, with the validation set containing 2318 data points. The model has been developed in 2 steps. Firstly, the data has been subjected to dimensionality reduction and then a model is chosen for the data. This model is then tuned based on cross validation results. 

### Feature Selection 
Lasso regression has been used to select relevant features. The lasso penalty is determined using cross validation on the training data. Features with non-zero coeffiecients from the Lasso regression are retained in the training dataset while the rest are discarded. The feature selection process discards only 1 attribute and the resultant training data is shrunk to 26 features. 

### Model Selection 
The classification algorithm is chosen through the emperical performance of the classifier on the validation set. Ensemble based, tree based, multilayer perceptron based, support vector machine based and nearest neighbors based classifiers were evaluated, and RandomForest regressor was chosen as a base classifier make predictions on the model. This model was chosen for its relatively superior performance and computational efficiency. 
The base classifier was tuned using GridSearchCV with mean absolute error criterion. The base tree estimators were pruned with minimum splitting size as 70 data points and minimum leaf size as 40 data points. 15 base estimators were used to get a final results. 
Pruning, parameter selection using cross validation, out-sample testing has been used to prevent overfitting in the model. 


## Model testing

The model is retrained on the training data and predictions are made on the prediction set. Results are outputted in the specified format. 

## Implementation

This project uses the following packages within a Python 3.7 environment
1) Pandas
2) Numpy
3) Sklearn
4) Category-encoders
