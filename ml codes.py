# Import necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns  # For enhanced data visualization
import joblib  # For saving and loading trained models
import warnings  # To manage and ignore warnings

# Import specific modules from scikit-learn for machine learning tasks
from sklearn.model_selection import train_test_split  # To split data into training and testing sets
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For data preprocessing (encoding and scaling)
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # For model evaluation
from sklearn.linear_model import LogisticRegression  # Logistic Regression model
from sklearn.ensemble import RandomForestClassifier  # Random Forest model
from sklearn.svm import SVC  # Support Vector Machine model
from sklearn.naive_bayes import GaussianNB  # Naive Bayes model
from sklearn.tree import DecisionTreeClassifier  # Decision Tree model
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors model

# Import XGBoost for gradient boosting
from xgboost import XGBClassifier

# Import RandomOverSampler for handling imbalanced datasets
from imblearn.over_sampling import RandomOverSampler

# Ignore any warnings that might arise during execution
warnings.filterwarnings("ignore")

# Load the dataset from a CSV file
# Make sure to replace "your_dataset.csv" with the actual name of your dataset file
df = pd.read_csv("your_dataset.csv")

# Data cleaning and preprocessing
# Replace specific string values with numerical or standardized ones
df = df.replace({'yes': 1, 'no': 0, '?': 'Others', 'others': 'Others'})
# Remove outlier data points based on the 'result' column
df = df[df['result'] > -5]

# Function to categorize age into different groups
def convertAge(age):
    if age < 4:
        return 'Toddler'
    elif age < 12:
        return 'Kid'
    elif age < 18:
        return 'Teenager'
    elif age < 40:
        return 'Young'
    else:
        return 'Senior'

# Apply the convertAge function to create a new 'ageGroup' column
df['ageGroup'] = df['age'].apply(convertAge)

# Feature engineering to create new features from existing ones
def add_feature(data):
    # Create a 'sum_score' by summing up scores from A1 to A10
    data['sum_score'] = data.loc[:, 'A1_Score':'A10_Score'].sum(axis=1)
    # Create an 'ind' feature by combining other binary features
    data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']
    return data

# Apply the feature engineering function to the dataframe
df = add_feature(df)
# Apply a log transformation to the 'age' column to handle skewness
df['age'] = df['age'].apply(lambda x: np.log(x))

# Encode categorical features into numerical values
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data

# Apply the encoding function to the dataframe
df = encode_labels(df)

# Prepare data for training
# Define features to be removed from the feature set (X)
removal = ['ID', 'age_desc', 'used_app_before', 'austim']
# Separate features (X) and the target variable (y)
X = df.drop(removal + ['Class/ASD'], axis=1)
y = df['Class/ASD']
# Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)

# Handle imbalanced data using RandomOverSampler
# This technique increases the number of instances in the minority class
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Normalize the features using StandardScaler
# This scales the data to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)
X_val = scaler.transform(X_val)

# Define a list of machine learning models to be trained
models = [
    ("Logistic Regression", LogisticRegression()),
    ("Random Forest", RandomForestClassifier()),
    ("SVM", SVC(kernel='rbf', probability=True)),
    ("Naive Bayes", GaussianNB()),
    ("Decision Tree", DecisionTreeClassifier()),
    ("KNN", KNeighborsClassifier()),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

# Train, evaluate, and save each model
for name, model in models:
    # Train the model on the resampled training data
    model.fit(X_resampled, y_resampled)
    # Make predictions on the validation set
    y_pred = model.predict(X_val)
    # Get prediction probabilities for ROC curve (if available)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else np.zeros_like(y_pred)

    # Print model performance metrics
    print(f"\n{name}")
    print(f"Training AUC: {roc_auc_score(y_resampled, model.predict(X_resampled)):.4f}")
    print(f"Validation AUC: {roc_auc_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred))

    # Save the trained model to a file
    joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")

    # Plot the Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    # Plot the ROC Curve if the model has the predict_proba method
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_val, y_proba):.2f})")

# Finalize and show the combined ROC curve plot for all models
plt.plot([0, 1], [0, 1], 'k--')  # Add the diagonal line for reference
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of All Models")
plt.legend()
plt.show()
