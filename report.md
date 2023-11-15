# Module 12 Report

## Overview of the Analysis

The main purpose of this report is to analyse the predictive power of two models to identify the creditworthiness of borrowers. 
For this, I used the file "lending_data.csv", in the Resources folder, historical lending activity from a peer-to-peer lending services company, and which has information of the loan status, and if it's considered a "Healthy Loan", or a  "High-Risk Loan".

MACHINE LEARNING PROCESS:
* After importing the dataset, it was separated in labels and features, and having a look at the "value_counts" of the labels, we can observe there are much less "High-Risk Loans" cases than "Healthy Loans".
* Then, I used train_test_split to randomly split the data in a training and a test subsets.
* To create the first model, I created a logistic regression model, with "lbfgs" as solver, and a random state of 1. I used the X_train, and y_train to fit the model, and then, the X_test to predict the label and compare it with the y_test actual values.
* After that, from sklearn.metrics, I used accuracy_scores, confusion_matrix and confusion_matrix to get the metrics of the model, which are described in the Results segment.
* Then, I created the second model using the RandomOverSampler module to resample the X_train, and the y_train, as X_res, and y_train to get a dataset with the same amount of "High-Risk Loans" and "Healthy Loans".
* Once having a dataset resampled, I did another logistic regression model, wich I fitted with X_res, and y_train, and then did the same process used in the first model.

## Results

* Machine Learning Model 1 (Just logistic regression):

  * The accuracy is really good, having a total of 99% of precision at predicting the classification of the type of loan.

  * In the precision section of the classification report, we observe that for the Healthy Loans, there is no problem, as it predicts 100% of the cases correctly, but the High-Risk Loans has a 87% of precision, meaning that 87% of the predicted high-risk loans were correct.

  * In the recall section, we have the same case, the Healthy Loans have no problem, but the High-Risk Loans have a recall of 89%, meaning that it's missing a 11% of the actual High-Risk Loans.




* Machine Learning Model 2 (Resampled data logistic regression):

  * The accuracy of this model, also has a high predictive value, a bit higher than the first one.

  * As in the previous model, we observe that for the Healthy Loans, there is no problem, predicting 100% of the cases correctly, also, the High-Risk Loans has a 87% of precision, meaning that 87% of the predicted high-risk loans were correct.

  * But in the recall section, we have a difference between the models, the Healthy Loans remains at 100%, and in the case of High-Risk Loans the recall value goes up to 100%, meaning it will predict as High-Risk Loans all the actual High-Risk Loans, but having some false positives.

## Summary

* Both of the models have a good accuracy, but for our purpose (predict possible risky loans), the first one doesn't seem that good, as it will fail to classify around 11% of the cases that could be a Risky Loan. So, in conclusion, even though the second model has the same predictive value for the High Risk Loans that the first one, the recall value indicates that the model can classify accurately 100% of the High Risk Loan cases. Maybe some of the Healthy Loans are classified as High Risk, but it seems better to loose a potential client, than a high amount of money in a Risky Loan, so, I would recomend to use the second model.
