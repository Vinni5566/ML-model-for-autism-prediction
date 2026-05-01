# ML Model for Autism Prediction

This project focuses on predicting the likelihood of Autism Spectrum Disorder (ASD) using behavioral and demographic data. To achieve this, we employ and compare seven different machine learning algorithms:

*   **Logistic Regression:** A foundational statistical model for binary classification.
*   **Random Forest:** An ensemble method that builds multiple decision trees to improve accuracy.
*   **Support Vector Machine (SVM):** A powerful model for finding the optimal hyperplane to separate classes.
*   **Naive Bayes:** A probabilistic classifier based on Bayes' theorem.
*   **Decision Tree:** A simple yet effective model that uses a tree-like structure for decision-making.
*   **K-Nearest Neighbors (KNN):** A non-parametric algorithm that classifies based on the majority class of its neighbors.
*   **XGBoost:** A highly efficient and scalable gradient boosting library.

The goal is to identify the most effective model for this prediction task.

## Key Features

*   **Data Preprocessing:** Includes label encoding, feature engineering, and outlier removal to prepare the data for modeling.
*   **Imbalanced Data Handling:** Addresses the class imbalance in the dataset using oversampling techniques.
*   **Multi-Model Training:** Trains and evaluates all seven machine learning models for a comprehensive comparison.
*   **Model Persistence:** Saves the trained models for future use and deployment.
*   **Comprehensive Evaluation:** Generates visualizations like ROC curves and confusion matrices to compare model performance.

## Tech Stack

*   **Python:** The core programming language for the project.
*   **Pandas & NumPy:** For efficient data manipulation and numerical operations.
*   **Scikit-learn:** A comprehensive library for machine learning in Python.
*   **XGBoost:** A specialized library for gradient boosting.
*   **imbalanced-learn:** A library to handle imbalanced datasets.
*   **Matplotlib & Seaborn:** For creating insightful data visualizations.

## How to Use

1.  **Dataset:** Ensure your dataset is in a `.csv` format and uploaded to the environment (e.g., Google Colab).
2.  **Preprocessing:** The script automatically preprocesses and cleans the data.
3.  **Model Training:** The seven models are trained, evaluated, and compared.
4.  **Results:** Performance metrics and visualizations are displayed to help you analyze the results.

This structured approach allows for a thorough evaluation of different models, helping you understand which one is best suited for predicting ASD based on the available data.