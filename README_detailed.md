# Predictive Analysis of Soil Types

## Introduction

This project aims to perform a predictive analysis on the "Soil Data Grevena" dataset. The primary goal is to classify soil types based on their chemical and physical properties. The dataset provides a rich set of features for this task, but lacks a pre-defined target variable for soil type. Therefore, this project demonstrates an end-to-end workflow that includes:

1.  **Data Loading and Pre-processing:** Loading the data from an Excel file and preparing it for analysis.
2.  **Exploratory Data Analysis (EDA):** Gaining insights into the data through visualization.
3.  **Synthetic Target Variable Generation:** Using a clustering algorithm to create a synthetic target variable for soil type.
4.  **Model Building:** Training and evaluating two different classification models to predict the generated soil types.
5.  **Clustering Analysis:** Applying a clustering algorithm to group the data based on its features.

This project serves as a comprehensive example of how to approach a predictive modeling problem, from data preparation to model evaluation and interpretation.

## Problem Statement

The objective of this project is to develop a model that can accurately predict the type of soil based on its chemical and physical properties. Since the original dataset does not contain a "Soil Type" column, the first step is to create a synthetic target variable by clustering the data. The subsequent problem is to train classification models on this synthetic data and evaluate their performance.

## Methodology / Approach

The analysis was conducted in the following steps:

### 1. Data Loading and Pre-processing

-   **Data Loading:** The dataset was loaded from the `SOIL DATA GR.xlsx` file into a pandas DataFrame.
-   **Column Name Cleaning:** All column names were stripped of leading and trailing whitespaces to prevent any potential errors during data manipulation.
-   **Missing Value Imputation:** The dataset was checked for missing values. It was found that the `Mn ppm` column had one missing value. To maintain data integrity, the row containing the missing value was dropped.
-   **Outlier Detection and Removal:** Outliers can significantly impact the performance of machine learning models. In this project, outliers were detected using the Z-score method. Any data point with a Z-score greater than 3 was considered an outlier and removed from the dataset.

### 2. Synthetic Target Variable Generation

-   **KMeans Clustering:** Since the `Soil Type` was not provided, the KMeans clustering algorithm was used to create a synthetic target variable. The data was grouped into three clusters based on the following features: `pH`, `EC mS/cm`, `O.M. %`, `N_NO3 ppm`, `P ppm`, `K ppm`, `Sand %`, `Silt %`, and `Clay %`.
-   **Label Encoding:** The cluster labels (0, 1, and 2) were then used as the `Soil Type` for the classification task. These labels were encoded into a numerical format suitable for the machine learning models.

### 3. Exploratory Data Analysis (EDA)

-   **Distribution of Soil Types:** A bar chart was created to visualize the distribution of the synthetically generated soil types.
-   **Feature Correlation:** A heatmap was generated to visualize the correlation between the different soil properties. This helps in understanding the relationships between the features.
-   **Pairplot:** A pairplot was created to visualize the pairwise relationships between the features, colored by the synthetic soil type.

### 4. Model Building

-   **Train-Test Split:** The dataset was split into training and testing sets, with 80% of the data used for training and 20% for testing.
-   **Feature Scaling:** The features were scaled using `StandardScaler` to ensure that all features have a mean of 0 and a standard deviation of 1. This is important for algorithms that are sensitive to the scale of the input features, such as Logistic Regression.
-   **Classification Models:** Two classification models were trained and evaluated:
    1.  **Random Forest Classifier:** An ensemble model that is known for its high accuracy and robustness.
    2.  **Logistic Regression:** A linear model that is simple to implement and interpret.
-   **Clustering Model:**
    1.  **KMeans Clustering:** The KMeans algorithm was used to group the data into clusters based on their features.

## Results / Snapshot of output

The models were evaluated on the test set, and the following results were obtained:

### Classification Results

-   **Random Forest Classifier:**
    -   **Accuracy:** 1.0
    -   **Classification Report:**

| Soil Type | Precision | Recall | F1-Score | Support |
| :-------- | :-------- | :----- | :------- | :------ |
| 0         | 1.00      | 1.00   | 1.00     | 55      |
| 1         | 1.00      | 1.00   | 1.00     | 4       |
| 2         | 1.00      | 1.00   | 1.00     | 86      |

-   **Logistic Regression:**
    -   **Accuracy:** 0.9586
    -   **Classification Report:**

| Soil Type | Precision | Recall | F1-Score | Support |
| :-------- | :-------- | :----- | :------- | :------ |
| 0         | 0.95      | 0.95   | 0.95     | 55      |
| 1         | 1.00      | 0.75   | 0.86     | 4       |
| 2         | 0.97      | 0.98   | 0.97     | 86      |

### Clustering Results

-   **KMeans Clustering:**

| Soil Type \ Cluster | 0  | 1   | 2   |
| :------------------ | :- | :-- | :-- |
| 0                   | 85 | 30  | 159 |
| 1                   | 10 | 0   | 12  |
| 2                   | 94 | 233 | 101 |

### Snapshots of Output

*Feature Importance from Random Forest*

![Feature Importance](https://i.imgur.com/YOUR_IMAGE_URL.png)

*KMeans Clusters*

![KMeans Clusters](https://i.imgur.com/YOUR_IMAGE_URL.png)

## Conclusion

This project successfully demonstrated a complete workflow for a predictive analysis task. The key takeaways are:

-   **Data pre-processing is crucial:** Cleaning the data, handling missing values, and dealing with outliers are essential steps for building a robust model.
-   **Synthetic data can be useful:** In the absence of a target variable, clustering can be a powerful technique to generate synthetic labels for a classification task.
-   **Model selection is important:** The Random Forest model achieved a perfect accuracy score, which is a strong indication of overfitting. The Logistic Regression model, while slightly less accurate, may be a more generalizable model. Further investigation, such as cross-validation, would be needed to confirm this.
-   **Interpretation is key:** The high accuracy of the models should be interpreted with caution, as the target variable was synthetically generated from the same features used for training. In a real-world scenario, it would be important to validate the generated clusters with a domain expert.

Overall, this project provides a solid foundation for further analysis of this dataset. Future work could involve exploring different clustering algorithms, using more advanced classification models, and validating the results with a domain expert.