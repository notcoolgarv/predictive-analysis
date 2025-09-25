# Soil Type Prediction using Soil Data Grevena

## 1. Domain
**Soil Types** — Predicting soil type based on chemical and physical properties.

## 2. Dataset
- **Source:** [Kaggle - Soil Data Grevena](https://www.kaggle.com/datasets/jocelyndumlao/soil-data-grevena)
- **File:** `SOIL DATA GR.xlsx`
- **Attributes:** The dataset contains various chemical and physical properties of soil samples, such as pH, Electrical Conductivity, and the percentage of Sand, Silt, and Clay.
- **Target Variable:** The `Soil Type` column was not present in the original dataset. For the purpose of this analysis, a synthetic `Soil Type` column was generated using the KMeans clustering algorithm. This allows us to demonstrate a complete classification workflow.

## 3. Problem Statement
Given the chemical and physical properties of soil, predict the soil type. This project tackles this as a multi-class classification problem.

## 4. Methodology

### a. Data Loading & Preprocessing
- **Data Loading:** The data was loaded from the `SOIL DATA GR.xlsx` Excel file using the pandas library.
- **Handling Missing Values:** Rows with any missing values were dropped to ensure data quality.
- **Column Name Cleaning:** Leading and trailing whitespaces were stripped from all column names to prevent potential errors during data manipulation.
- **Synthetic Target Variable:** As the original dataset lacks a target variable for soil type, the KMeans clustering algorithm was employed to create a synthetic `Soil Type` column. The data was grouped into three clusters, which were then used as the target labels for the classification task.
- **Label Encoding:** The categorical `Soil Type` labels ('0', '1', '2') were converted into a numerical format using `LabelEncoder` from scikit-learn.
- **Feature Scaling:** All numerical features were scaled using `StandardScaler` to standardize the data and improve the performance of the machine learning models.
- **Outlier Detection:** Outliers were detected using the z-score method. Data points with a z-score greater than 3 were identified as outliers and removed from the dataset.

### b. Exploratory Data Analysis (EDA)
- **Distribution of Soil Types:** A countplot was generated to visualize the distribution of the synthetically generated soil types.
- **Feature Correlation:** A correlation heatmap was created to explore the relationships between the different soil properties.
- **Pairplot:** A pairplot was used to visualize the relationships between pairs of features, with the points colored by the synthetic soil type.

### c. Model Building
- **Classification:**
  - Two classification algorithms were applied:
    - **Random Forest Classifier:** An ensemble learning method that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes of the individual trees.
    - **Logistic Regression:** A linear model for classification rather than regression.
  - The models were evaluated using the following metrics:
    - **Accuracy:** The proportion of correctly classified samples.
    - **Classification Report:** A detailed report showing the precision, recall, and F1-score for each class.
    - **Confusion Matrix:** A table used to describe the performance of a classification model on a set of test data for which the true values are known.
- **Clustering:**
  - **KMeans Clustering:** The KMeans algorithm was applied to the data to group it into clusters based on their features. The results were presented in a cross-tabulation against the synthetic soil types.

### d. Results
- **Random Forest Classifier:** Achieved an accuracy of 1.0. While this is a perfect score, it is highly likely that the model is overfitting the data. This is a common issue with tree-based models, especially when the target variable is generated from the same features used for training.
- **Logistic Regression Classifier:** Achieved an accuracy of approximately 0.96, which is a very good result, but may also be influenced by the synthetic nature of the target variable.
- **KMeans Clustering:** The algorithm successfully grouped the data into three distinct clusters, which were then used as the basis for the classification task.

## 5. How to Run

1. **Download the Dataset:** Download the `SOIL DATA GR.xlsx` file from the [Kaggle dataset page](https://www.kaggle.com/datasets/jocelyndumlao/soil-data-grevena) and place it in the same directory as the script.
2. **Install Dependencies:** Make sure you have Python installed, and then install the required libraries using pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
   ```
3. **Run the Script:** Execute the following command in your terminal:
   ```bash
   python soil_type_prediction.py
   ```

## 6. Conclusion
This project demonstrates a complete predictive analysis workflow, from data loading and preprocessing to model building and evaluation. While the use of a synthetically generated target variable limits the real-world applicability of the classification results, the project serves as a comprehensive example of how to approach a data science problem when faced with an unsupervised learning task that can be framed as a supervised one. The high accuracy scores, while promising, should be interpreted with caution due to the nature of the target variable's creation.