# Milk Data Machine Learning Analysis

Implemented and fine-tuned supervised machine learning models in Python, leveraging Scikit-Learn to develop a robust predictive model for assessing the quality of milk. Models: Decision Tree, Random Forest, Multinomial Logistic Regression, Linear Discriminant Analysis, and K Nearest Neighbours (KNN)

Data Preprocessing:
- Analyzed the shape and the contents of the data such as null values
- Converted the grade column values from low/medium/high to 0/1/2
- Identified the correlation between the predictors by creating a correlation matrix

Data Visualization:
- Created a correlation heatmap 
- Created a barplot using the altair library to visualize the records for the grade variable and ran variance inflation factor to determine correlations between more than 2 predictors. 

Data Standardization and Splitting:
- Scaled/Standardized the data using the scikitlearn library
- Split the data into a 80/20 training/testing split

For each model:
- Determined the best parameters using grid search cross validation
- Printed the accuracy 
- Created a classification report
- Created a confusion matrix to visualize the number of hits and misses

-Compared the accuracy of each model and deployed the one with the highest accuracy
