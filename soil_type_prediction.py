import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from scipy.stats import zscore
import os


def load_data(filepath):
    """
    Load data from a CSV or Excel file.
    """
    # Support both CSV and Excel
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)
    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    return df

def preprocess_data(df):
    """
    Preprocess the data by handling missing values, creating a target variable if it doesn't exist,
    encoding categorical features, and scaling the data.
    """
    # Drop duplicates
    df = df.drop_duplicates()
    # Drop rows with missing values
    df = df.dropna()
    # --- Fix: Check for 'Soil Type' column or create it if missing ---
    # If 'Soil Type' does not exist, create it using a rule or clustering
    if 'Soil Type' not in df.columns:
        print("Column 'Soil Type' not found. Creating synthetic soil types using KMeans clustering for demonstration.")
        # Use KMeans to create 3 clusters as soil types (you can change n_clusters as needed)
        features_for_clustering = ['pH', 'EC mS/cm', 'O.M. %', 'N_NO3 ppm', 'P ppm', 'K ppm', 'Sand %', 'Silt %', 'Clay %']
        # Ensure all required columns exist
        for col in features_for_clustering:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in the dataset.")
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['Soil Type'] = kmeans.fit_predict(df[features_for_clustering])
        df['Soil Type'] = df['Soil Type'].astype(str)
    # Encode categorical target
    le = LabelEncoder()
    df.loc[:, 'Soil Type Encoded'] = le.fit_transform(df['Soil Type'])
    # Feature scaling
    # Use correct feature names from your Excel
    features = ['pH', 'EC mS/cm', 'O.M. %', 'N_NO3 ppm', 'P ppm', 'K ppm', 'Sand %', 'Silt %', 'Clay %']
    # Ensure all required columns exist
    for col in features:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the dataset.")
    scaler = StandardScaler()
    df.loc[:, features] = scaler.fit_transform(df[features])
    return df, le, features

def plot_eda(df, features):
    """
    Perform exploratory data analysis by plotting the distribution of soil types,
    a correlation heatmap, and a pairplot.
    """
    plt.figure(figsize=(10,6))
    sns.countplot(x='Soil Type', data=df)
    plt.title('Distribution of Soil Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(df[features + ['Soil Type Encoded']].corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()

    # Pairplot
    sns.pairplot(df, hue='Soil Type', vars=features[:6])
    plt.show()

def detect_outliers(df, features):
    """
    Detect and remove outliers using the z-score.
    """
    # Simple outlier detection using z-score
    z_scores = np.abs(zscore(df[features]))
    outliers = (z_scores > 3).any(axis=1)
    print(f"Number of outliers detected: {outliers.sum()}")
    # Optionally, remove outliers
    df_clean = df[~outliers].copy()
    return df_clean

def train_classifiers(X_train, X_test, y_train, y_test, le, features):
    """
    Train and evaluate two classification algorithms: Random Forest and Logistic Regression.
    """
    # Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    print("Random Forest Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    # Feature importance
    importances = clf_rf.feature_importances_
    plt.figure(figsize=(8,5))
    sns.barplot(x=importances, y=features)
    plt.title('Random Forest Feature Importances')
    plt.tight_layout()
    plt.show()

    # Logistic Regression
    clf_lr = LogisticRegression(max_iter=200)
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_test)
    print("\nLogistic Regression Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr, target_names=le.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

def clustering_analysis(X, df, features):
    """
    Perform KMeans clustering and visualize the results.
    """
    # KMeans clustering
    kmeans = KMeans(n_clusters=df['Soil Type Encoded'].nunique(), random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    df = df.copy()
    df['Cluster'] = clusters
    print("\nKMeans Clustering Results:")
    print(pd.crosstab(df['Soil Type'], df['Cluster']))
    # Visualize clusters using first two features
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=features[0], y=features[1], hue='Cluster', data=df, palette='Set1')
    plt.title('KMeans Clusters (using first two features)')
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the analysis.
    """
    # Change the filename to your Excel file
    df = load_data('SOIL DATA GR.xlsx')
    print("First 5 rows:\n", df.head())
    print("\nInfo:\n")
    df.info()
    print("\nMissing values:\n", df.isnull().sum())
    print("\nColumns:\n", df.columns)

    # Preprocessing
    df, le, features = preprocess_data(df)
    # Plot EDA
    plot_eda(df, features)
    # Detect and remove outliers
    df = detect_outliers(df, features)

    # Prepare data for ML
    X = df[features]
    y = df['Soil Type Encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Classification
    train_classifiers(X_train, X_test, y_train, y_test, le, features)

    # Clustering
    clustering_analysis(X, df, features)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()

