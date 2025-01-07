Emotion Recognition Using Echocardiogram Data

This repository contains a project aimed at exploring the feasibility of recognizing emotions from echocardiogram (not electrocardiogram or ECG) data. While this idea comes with inherent challenges due to the lack of emotional context in echocardiogram datasets, a theoretical model has been implemented.

Background

The project originated from an assignment where the objective was to classify emotions based on echocardiogram data. This task is particularly challenging because:

Echocardiogram data primarily focuses on cardiac structure and function, providing limited emotional context.

Extracting meaningful features from echocardiogram videos would typically require advanced techniques like 3D Convolutional Neural Networks (CNNs) or LSTM-transformers.

Despite these challenges, this project demonstrates a working pipeline using synthetic emotional labels derived from available echocardiogram datasets.

Dataset

The dataset used for this project is sourced from the UCI Machine Learning Repository's Echocardiogram Dataset. The dataset was selected due to its accessibility, though it provides no explicit emotional context. Synthetic labels were assigned to mimic emotional states (e.g., Stress, Calm, Excitement) based on specific echocardiogram features.

Methodology

1. Data Preprocessing

Data was cleaned to handle missing values and normalize continuous features.

Unnecessary columns were dropped to streamline the analysis.

2. Label Assignment

Synthetic labels were assigned based on heuristic conditions:

Stress: Low fractional shortening and high wall motion score.

Calm: Medium fractional shortening and low wall motion score.

Excitement: High fractional shortening and medium wall motion score.

3. Noise Addition

Random noise was added to continuous features to simulate variability and improve model robustness.

4. Model Training

Four machine learning models were implemented and evaluated:

Random Forest Classifier (with hyperparameter tuning using GridSearchCV)

Logistic Regression

Support Vector Machine (SVM)

k-Nearest Neighbors (k-NN)

5. Handling Class Imbalance

Synthetic Minority Oversampling Technique (SMOTE) was used to address class imbalance issues.

6. Evaluation

Models were evaluated using metrics such as precision, recall, F1-score, and confusion matrices. Visualization techniques were employed to illustrate feature distributions, decision boundaries, and classification results.

Results

The models demonstrated the feasibility of emotion classification using synthetic labels, though the results remain highly theoretical due to the limitations of the dataset and label assignment method.

Challenges

Dataset Limitation: The chosen dataset lacks inherent emotional information.

Feature Extraction Complexity: Advanced video analysis techniques were beyond the scope of this assignment.

Theoretical Nature: Synthetic labels and heuristic rules limit the practical applicability of the results.

Future Directions

To improve the robustness and applicability of this project:

Use datasets with explicit emotional context.

Employ advanced feature extraction techniques like 3D CNNs or LSTM-transformers for echocardiogram video data.

Collaborate with domain experts to design realistic labeling strategies.

How to Run

Clone the repository.

Install the required dependencies listed in requirements.txt.

Run the preprocessing and training scripts using the provided dataset.

Visualize the results through the generated plots and reports.

Acknowledgments

This project was completed independently to explore an unconventional application of machine learning. While the task was challenging and fraught with conceptual limitations, it serves as a learning opportunity and a demonstration of perseverance in the face of academic constraints.





