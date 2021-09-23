# What is driving the errors in the Zestimates?

## Project Overview

In this project, I will be working with a Zillow dataset to create a model that will discover what is driving error in zestimates 


## Project Description

We are to use clustering and regression models to help us discover drivers for logerrors in the zestimates. Since there are heaps of missing data, we'll have to find ways to combat it. 

## Goals

Deliver a Jupyter notebook going through the steps of the data science pipeline
Create clusters in uncovering what the drivers of the error in the zestimate is.
Discover features that contribute to logerror
Present a notebook about my findings

## Deliverables
Finalized Jupyter notebook complete with comments
A README.md with executive summary, contents, data dictionary, conclusion and next steps, and how to recreate this project.

## Project Summary
I incorporated clustering to discover keys drivers in logerror of zestimates using a Zillow data frame.

## Data Dictionary 

| Column Name                  | Renamed   | Info                                            |
|------------------------------|-----------|-------------------------------------------------|
| 
<br>

## Hypothesis 

1.) Log error is affected by square footage, number of bedrooms

2.) Log error is affected by square footage

3.) Log error is affected by longitude and lattitude

4.) Log error is affected by baths, square foot, ad tax amount

## Findings and Next Steps 
   - 


Next steps would be:
 - gather more information on location
 -  


# The Pipeline

## Planning 
Goal: Plan out the project
How does certain features effect the logerror

I also want to look into other features, like age and see if that will also correlate to property value. 
A lot of these features could play hand in hand and help my model make better predictions.

Hypotheses: Square footage, beds and bathrooms, and location will have an effect on the logerror


## Acquire 
Goal: Have Zillow dataframe ready to prepare in first part of wrangle.py In this stage, I used a connection URL to access the CodeUp database. Using a SQL query, I brought in the 2017 Zillow dataset with only properties set for single use, and joined them with other tables via parcelid to get all of their features. I turned it into a pandas dataframe and created a .csv in order to use it for the rest of the pipeline.

## Prep 
Goal: Have Zillow dataset that is split into train, validate, test, and ready to be analyzed. Assure data types are appropriate and that missing values/duplicates/outliers are addressed. Put this in our wrangle.py file as well. In this stage, I handled missing values by dropping any rows and columns with more than 50% missing data.

I assured that all columns had a numeric data type, and renamed them for ease of use.

Duplicates were dropped (in parcelid)

Nulls in square footage, lotsize, tax value, and tax amount were imputed with median. (after splitting)

Nulls in calculatedbathnbr, full bath count, region id city, regionidzip, and censustractandblock were imputed with most frequent. (after splitting)

Any remaining nulls after these were dropped. I split the data into train, validate, test, X_train, y_train, X_validate, y_validate, X_test, and y_test. Last, I scaled it on a StandardScaler scaler (I made sure to drop outliers first!) and also returned X_train, X_validate, and X_test scaled.

## Explore 
Goal: Visualize the data and explore relationships. Use the visuals and statistics tests to help answer your questions. 
I plotted distributions, made sure nothing was out of the ordinary after cleaning the dataset. 
Plotted a pairplot to see combinations of variables.
I ran t-tests with the features in respect to logerror. Also a few to see if the independent variables were related to each other. 


## Modeling and Evaluation 
Goal: Along with regression models, clusters were used to identify statistical relationship to logerror

The models worked best with xxxxxx. Polynomial Regression performed the best, so I did a test on it.

| Model                            | RMSE Training | RMSE Validate | R^2   |
|----------------------------------|---------------|---------------|-------|
| Baseline                         |     |    |  |
| LinearRegression                 |     |    |  |
| LassoLars                        |    |    |  |
| TweedieRegressor                 |    |     |  |
| PolynomialRegression (3 degrees) |     |     | |

Test:
 - RMSE of 
 - R^2 of 

## Delivery 
A final notebook walkthrough of the my findings will be given 
 - All acquire and prepare .py files are uploaded for easy replication.
 - This README 
 - Final notebook that documents a commented walkthrough of my process

# Conclusion 



# How to Recreate Project

 - You'll need your own username/pass/host credentials in order to use the get_connection function in my acquire.py to access the Zillow database
 - Have a copy of my acquire, prep, explore .py files. 
 - My final notebook has all of the steps outlined, and it is really easy to adjust parameters.
