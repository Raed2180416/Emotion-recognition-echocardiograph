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

# Step 4: Add Noise to Data
def add_noise(df, noise_level=0.05):
    noisy_df = df.copy()
    numeric_features = ["fractional_shortening", "wall_motion_score"]
    for feature in numeric_features:
        noise = noise_level * np.random.normal(size=df[feature].shape)
        noisy_df[feature] = df[feature] + noise
        noisy_df[feature] = noisy_df[feature].clip(0, 1)  # Ensure values stay within [0, 1]
    return noisy_df

# Step 5: Train and Evaluate Models
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
    cm = confusion_matrix(y_test, y_pred)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stress", "Calm", "Excitement"]))
    visualize_confusion_matrix(cm, "Random Forest")
    return model

def train_logistic_regression(X_train, X_test, y_train, y_test):
    # Train Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stress", "Calm", "Excitement"]))
    visualize_confusion_matrix(cm, "Logistic Regression")
    return model

def train_svm(X_train, X_test, y_train, y_test):
    # Train SVM
    model = SVC(kernel='rbf', random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stress", "Calm", "Excitement"]))
    visualize_confusion_matrix(cm, "SVM")
    return model

def train_knn(X_train, X_test, y_train, y_test):
    # Train k-NN
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("k-NN Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stress", "Calm", "Excitement"]))
    visualize_confusion_matrix(cm, "k-NN")
    return model

def cross_validation_score(model, X, y):
    # Perform k-fold cross-validation
    scores = cross_val_score(model, pd.DataFrame(X, columns=["fractional_shortening", "wall_motion_score"]), y, cv=5, scoring='f1_weighted')
    print(f"Cross-Validation F1-Scores: {scores}")
    print(f"Mean F1-Score: {np.mean(scores)}")

def feature_importance_analysis(model, feature_names):
    # Analyze feature importances for Random Forest
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for name, importance in zip(feature_names, importances):
            print(f"Feature: {name}, Importance: {importance:.4f}")
    else:
        print("Feature importance not available for this model.")

# Step 6: Visualization
def visualize_confusion_matrix(cm, model_name):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stress", "Calm", "Excitement"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.show()

def visualize_decision_boundaries(X, y, model, model_name):
    # Plot decision boundaries for models
    x_min, x_max = X.iloc[:, 0].min() - 0.1, X.iloc[:, 0].max() + 0.1
    y_min, y_max = X.iloc[:, 1].min() - 0.1, X.iloc[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = model.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=["fractional_shortening", "wall_motion_score"]))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title(f"Decision Boundary: {model_name}")
    plt.xlabel("fractional_shortening")
    plt.ylabel("wall_motion_score")
    plt.colorbar(scatter)
    plt.show()

# Main Execution
input_file = "echocardiogram.data"
output_file = "echocardiogram_cleaned.data"
preprocess_file(input_file, output_file)
df = load_and_preprocess_data(output_file)
df = assign_labels(df)

# Add noise to the dataset
df_noisy = add_noise(df)

# Prepare data for training
filtered_df = df_noisy[df_noisy["emotion"] != "Unknown"].copy()
filtered_df["emotion"] = filtered_df["emotion"].astype("category").cat.codes
X = filtered_df[["fractional_shortening", "wall_motion_score"]]
y = filtered_df["emotion"]

# Address class imbalance using SMOTE
k_neighbors = min(5, X[y == y.value_counts().idxmin()].shape[0] - 1)
if k_neighbors >= 1:
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    try:
        X, y = smote.fit_resample(X, y)
        X = pd.DataFrame(X, columns=["fractional_shortening", "wall_motion_score"])
    except ValueError as e:
        print(f"SMOTE failed: {e}")
        print("Continuing without SMOTE...")
else:
    print("Not enough samples for SMOTE. Continuing without oversampling...")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = pd.DataFrame(X_train, columns=["fractional_shortening", "wall_motion_score"])
X_test = pd.DataFrame(X_test, columns=["fractional_shortening", "wall_motion_score"])

# Train and evaluate models
rf_model = train_random_forest(X_train, X_test, y_train, y_test)
cross_validation_score(rf_model, X, y)
feature_importance_analysis(rf_model, ["fractional_shortening", "wall_motion_score"])
visualize_decision_boundaries(X, y, rf_model, "Random Forest")

log_reg_model = train_logistic_regression(X_train, X_test, y_train, y_test)
cross_validation_score(log_reg_model, X, y)
visualize_decision_boundaries(X, y, log_reg_model, "Logistic Regression")

svm_model = train_svm(X_train, X_test, y_train, y_test)
cross_validation_score(svm_model, X, y)
visualize_decision_boundaries(X, y, svm_model, "SVM")

knn_model = train_knn(X_train, X_test, y_train, y_test)
cross_validation_score(knn_model, X, y)
visualize_decision_boundaries(X, y, knn_model, "k-NN")

