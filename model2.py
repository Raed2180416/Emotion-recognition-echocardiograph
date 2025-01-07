import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Step 1: Preprocess the Data File
def preprocess_file(input_file, output_file):
    cleaned_data = []
    expected_columns = 13

    with open(input_file, "r") as file:
        for line in file:
            fields = line.strip().split(",")
            if len(fields) == expected_columns:
                cleaned_data.append(fields)
            else:
                print(f"Skipping line: {line.strip()} (expected {expected_columns} columns, got {len(fields)})")

    with open(output_file, "w") as file:
        for row in cleaned_data:
            file.write(",".join(row) + "\n")

# Step 2: Load and Preprocess Data
def load_and_preprocess_data(cleaned_file):
    columns = [
        "survival_months", "still_alive", "age", "pericardial_effusion",
        "fractional_shortening", "epss", "lvdd", "wall_motion_score",
        "wall_motion_index", "mult", "name", "group", "alive_at_1"
    ]
    df = pd.read_csv(cleaned_file, header=None, names=columns, na_values="?")

    # Drop unnecessary columns
    df.drop(columns=["mult", "name", "group"], inplace=True)

    # Handle missing values
    df.fillna(df.median(), inplace=True)

    # Normalize continuous features
    scaler = MinMaxScaler()
    numeric_features = ["fractional_shortening", "epss", "lvdd", "wall_motion_score", "wall_motion_index"]
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    return df

# Step 3: Assign Synthetic Emotional Labels
def assign_labels(df):
    conditions = [
        (df["fractional_shortening"] < 0.4) & (df["wall_motion_score"] > 0.6),
        (df["fractional_shortening"] >= 0.4) & (df["fractional_shortening"] <= 0.7) & (df["wall_motion_score"] <= 0.4),
        (df["fractional_shortening"] > 0.7) & (df["wall_motion_score"] <= 0.6),
    ]
    labels = ["Stress", "Calm", "Excitement"]
    df["emotion"] = np.select(conditions, labels, default="Unknown")
    return df

# Step 4: Train and Evaluate Models
def train_random_forest(X_train, X_test, y_train, y_test):
    # Hyperparameter tuning for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1_weighted')
    grid_search.fit(X_train, y_train)

    print("Best Parameters (Random Forest):", grid_search.best_params_)

    # Train Random Forest classifier with best parameters
    model = RandomForestClassifier(random_state=42, **grid_search.best_params_)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stress", "Calm", "Excitement"]))
    return model

def train_logistic_regression(X_train, X_test, y_train, y_test):
    # Train Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stress", "Calm", "Excitement"]))
    return model

def train_svm(X_train, X_test, y_train, y_test):
    # Train SVM
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stress", "Calm", "Excitement"]))
    return model

def train_knn(X_train, X_test, y_train, y_test):
    # Train k-NN
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("k-NN Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stress", "Calm", "Excitement"]))
    return model

# Step 5: Visualization
def visualize_results(df, cm):
    # Visualize feature distributions
    for feature in ["fractional_shortening", "wall_motion_score"]:
        plt.figure()
        for label in df["emotion"].unique():
            subset = df[df["emotion"] == label]
            plt.hist(subset[feature], alpha=0.5, label=label, bins=10)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.legend()
        plt.show()

    # Visualize confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stress", "Calm", "Excitement"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# Main Execution
input_file = "echocardiogram.data"
output_file = "echocardiogram_cleaned.data"
preprocess_file(input_file, output_file)
df = load_and_preprocess_data(output_file)
df = assign_labels(df)

# Prepare data for training
filtered_df = df[df["emotion"] != "Unknown"].copy()
filtered_df["emotion"] = filtered_df["emotion"].astype("category").cat.codes
X = filtered_df[["fractional_shortening", "wall_motion_score"]]
y = filtered_df["emotion"]

# Address class imbalance using SMOTE
k_neighbors = min(5, X[y == y.value_counts().idxmin()].shape[0] - 1)
if k_neighbors >= 1:
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    try:
        X, y = smote.fit_resample(X, y)
    except ValueError as e:
        print(f"SMOTE failed: {e}")
        print("Continuing without SMOTE...")
else:
    print("Not enough samples for SMOTE. Continuing without oversampling...")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate models
rf_model = train_random_forest(X_train, X_test, y_train, y_test)
log_reg_model = train_logistic_regression(X_train, X_test, y_train, y_test)
svm_model = train_svm(X_train, X_test, y_train, y_test)
knn_model = train_knn(X_train, X_test, y_train, y_test)
