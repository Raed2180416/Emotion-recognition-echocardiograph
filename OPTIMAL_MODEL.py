import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load and preprocess dataset
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
    numeric_features = ["fractional_shortening", "wall_motion_score"]
    for feature in numeric_features:
        df[feature] = (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())

    return df

# Assign synthetic labels
def assign_labels(df):
    conditions = [
        (df["fractional_shortening"] < 0.4) & (df["wall_motion_score"] > 0.6),
        (df["fractional_shortening"] >= 0.4) & (df["fractional_shortening"] <= 0.7) & (df["wall_motion_score"] <= 0.4),
        (df["fractional_shortening"] > 0.7) & (df["wall_motion_score"] <= 0.6),
    ]
    labels = ["Stress", "Calm", "Excitement"]
    df["emotion"] = np.select(conditions, labels, default="Unknown")
    return df

# Train Random Forest and display feature importance and decision boundary
def train_random_forest_with_visualizations(df):
    # Filter data with valid labels
    df = df[df["emotion"] != "Unknown"].copy()
    df["emotion"] = df["emotion"].astype("category").cat.codes

    # Features and target
    X = df[["fractional_shortening", "wall_motion_score"]]
    y = df["emotion"]

    # Adjust k_neighbors for SMOTE dynamically
    class_counts = y.value_counts()
    min_class_samples = class_counts.min()
    k_neighbors = min(5, min_class_samples - 1)  # Ensure k_neighbors is valid

    # Apply SMOTE for class imbalance
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=2, random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Stress", "Calm", "Excitement"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stress", "Calm", "Excitement"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    # Feature Importance
    feature_importances = model.feature_importances_
    features = ["fractional_shortening", "wall_motion_score"]
    print("Feature Importances:")
    for feature, importance in zip(features, feature_importances):
        print(f"Feature: {feature}, Importance: {importance:.4f}")

    # Visualize Feature Importances
    plt.bar(features, feature_importances, color='skyblue')
    plt.title("Feature Importances")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.show()

    # Visualize Decision Boundary
    x_min, x_max = X_resampled.iloc[:, 0].min() - 0.1, X_resampled.iloc[:, 0].max() + 0.1
    y_min, y_max = X_resampled.iloc[:, 1].min() - 0.1, X_resampled.iloc[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    # Wrap grid points into a DataFrame with correct feature names
    grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=["fractional_shortening", "wall_motion_score"])
    Z = model.predict(grid_points).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    scatter = plt.scatter(X_resampled.iloc[:, 0], X_resampled.iloc[:, 1], c=y_resampled, edgecolor='k', cmap=plt.cm.coolwarm)
    plt.title("Decision Boundary")
    plt.xlabel("fractional_shortening")
    plt.ylabel("wall_motion_score")
    plt.colorbar(scatter)
    plt.show()

    return model

# Main Execution
cleaned_file = "echocardiogram_cleaned.data"
df = load_and_preprocess_data(cleaned_file)
df = assign_labels(df)
rf_model = train_random_forest_with_visualizations(df)



