# Credit Card Default Risk Analysis

## Overview

This project aims to leverage data analytics and machine learning techniques to analyze and predict credit card default risk. A financial service company's primary product is a high-value credit card associated with high customer lifetime values (LTVs). The partner is looking to reduce losses from customer defaults by offering a new, lower-value card that supports balance management and helps customers avoid default. By proactively identifying customers at risk of defaulting on credit card payments, the goal is to minimize financial losses and enhance customer relationship management for financial institutions. These customers who are at risk of defaulting on credit card will be offered the new balance management card while keeping the good customers in the high-LTV traditional card.

## Motivation

Predicting credit card default risk is crucial for financial institutions as it helps in mitigating financial risks and understanding customer behavior. This understanding enables the development of tailored financial solutions that foster customer loyalty and trust.

## Dataset

The dataset encompasses various aspects of customer information, including:

- **Demographic Information**: Gender, education level, marital status, and age.
- **Credit Information**: Credit limit, indicating the institution's trust in the customer.
- **Payment History**: Detailed records from the past six months, offering insights into the customer's financial reliability.
- **Bill Amounts & Payment Amounts**: Monthly billing statements and actual payments made, shedding light on financial behavior and liquidity.
- **Default Status**: The target variable indicating whether a customer defaulted in October.

The dataset is split into a training set (80%) for model training and a test set (20%) for predictions.

## Exploratory Data Analysis (EDA)

### Missing Values

An extensive EDA was conducted to clean the data, handle missing values, and identify anomalies. The approach was data-driven, aiming to understand the underlying patterns that might influence default risks.

The analysis of missing values showed that their distribution is random and does not differ significantly between features with and without missing values. Three different imputation techniques were compared: `KNN`, `Iterative Imputer`, and `Median Imputer`. For categorical features, missing values were imputed using the most frequent (mode) values.

### Outliers

The following columns were identified as having outliers in the dataset:

- `pay_1` to `pay_6`: Valid values are -1 and above, but minimum values of -2 were present.
- `education`: Valid values are 1-4, but minimum values of 0 and maximum values of 6 were found.
- `marriage`: Valid values are 1-3, but a minimum value of 0 was present.

Since these outliers are associated with categorical columns, they were treated as separate categories for their respective features.

## Model Building

### Base Model

The initial model used was Logistic Regression, which served as the base model. A pipeline was developed to handle standardization of numerical features, one-hot encoding of categorical features, and SMOTE (Synthetic Minority Over-sampling Technique) to address the imbalanced dataset. `Log loss` was used as the evaluation metric to compare various modeling approaches. The base model achieved a log loss score of `0.58`.

### Missing Value Imputation

To further improve model performance, three different approaches for imputing missing values were compared: `KNN, Iterative Imputer, and Median Imputer`. Hyperparameter tuning was also incorporated for the Logistic Regression model. The log loss score for all three approaches was `0.56`, indicating that missing values do not significantly influence the overall model performance. Since the imputation methodology did not impact the score substantially, the `Median Imputer` was chosen for subsequent steps.

### Ensemble Modeling

Ensemble models were built using `Logistic Regression, Random Forest, and Gradient Boosting`. Each of these models was hypertuned using two different approaches: `Randomized Search and Optuna`. The log loss scores of the hypertuned models were compared, and the best model for each algorithm was chosen to build an ensemble model.

Two ensemble strategies were employed: `Voting Classifier and Stacking Classifier`. The log loss scores for the Voting Classifier and Stacking Classifier with Randomized Search were `0.48 and 0.43`, respectively, while the scores with Optuna were `0.47 and 0.43`.

### Clustering

Clustering techniques, namely `KNN` and `DBSCAN`, were explored to identify potential clusters that could be used as additional features in the model. However, both approaches yielded only a `single` cluster, which did not provide additional information.

### Feature Engineering

Feature engineering played a crucial role in improving the model's performance. The following features were engineered to capture more meaningful insights from the data:

- **Demographic Interactions**: A new feature called `edu_marriage_interaction` was created by combining the `education` and `marriage` columns. This interaction feature aimed to capture the relationship between education level and marital status, which could influence default risk.


- **Payment-to-Bill Ratios**: For each of the past six months, a new feature was created to calculate the ratio of the payment amount to the bill amount. These features, named `pay_bill_ratio_1` to `pay_bill_ratio_6`, provided insights into the customer's payment behavior and ability to meet financial obligations.


- **Aggregated Payment and Bill Features**: Three new features were added to capture the overall payment and bill behavior: `total_bill_amt`, `total_pay_amt`, and `bill_pay_ratio`. These features provided a holistic view of the customer's financial situation and payment habits.


- **Age Group Binning**: The `age` feature was binned into groups to capture potential non-linear relationships between age and default risk. The new feature, `age_group`, categorized customers into six age groups: '20-29', '30-39', '40-49', '50-59', '60-69', and '70-79'.


These feature engineering techniques aimed to capture more meaningful information from the data, potentially improving the model's ability to predict credit card default risk.

## Conclusion

This project successfully leveraged data analytics and machine learning techniques to analyze and predict credit card default risk