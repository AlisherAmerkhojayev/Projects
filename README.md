**🧠 Churn Prediction Using Decision Tree Ensembles and Naïve Bayes in MATLAB**

📌 Overview
This project investigates customer churn prediction using supervised machine learning models: Naïve Bayes (NB) and Decision Tree Ensembles (DTE). We utilize SMOTE to address class imbalance and perform hyperparameter tuning for optimal model performance.

The analysis is based on the Churn Modelling dataset (10,000 records) from Kaggle, representing banking customer data. We compare the strengths, weaknesses, and real-world applicability of both algorithms using a robust evaluation pipeline.

🎯 Objectives
Predict whether a customer is likely to leave a bank.

Compare performance between Naïve Bayes and Decision Tree Ensembles.

Mitigate data imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

Optimize model parameters via grid search and cross-validation.

Evaluate results using metrics such as Accuracy, Precision, Recall, F1-score, and AUC.

🧪 Dataset & Features
Source: Kaggle - Churn Modelling Dataset

Size: 10,000 rows × 14 columns

Target Variable: Exited (1 = churned, 0 = retained)

Features used:

Numeric: CreditScore, Age, Tenure, Balance, NumOfProducts, EstimatedSalary, etc.

Categorical: Geography, Gender (converted to numeric using grp2idx)

Class imbalance present: ~20% churned vs. ~80% retained.

🛠️ Methodology
1. Data Preprocessing
Encoded categorical variables.

Selected relevant features.

Split dataset into 80% training and 20% testing.

2. SMOTE
Applied SMOTE to synthetically oversample the minority class (churned customers).

Used k=5 nearest neighbors to generate synthetic data.

3. Naïve Bayes Model
Trained using fitcnb with normal distribution assumptions.

Fast and computationally efficient.

Used as a baseline for model performance.

4. Decision Tree Ensemble (Bagging)
Trained using fitcensemble with method set to 'Bag'.

Hyperparameters tuned:

MaxNumSplits: [1–10]

MinLeafSize: [1–10]

MinParentSize: [10–100]

Evaluated performance on validation set using AUC.

Selected best configuration for final testing.

5. Model Evaluation
Compared models using:

Confusion Matrix

Accuracy, Precision, Recall, F1-score

ROC-AUC

Visual comparison of ROC curves

📊 Results Summary
Metric	Naïve Bayes	Decision Tree Ensemble
Accuracy	57.2%	76.6%
Precision	0.93	0.90
Recall	0.51	0.79
F1-score	0.636	0.84
AUC	0.78	0.81

Naïve Bayes:

High precision but low recall.

Underperforms on detecting churners.

Fast and lightweight.

Decision Tree Ensemble:

Superior overall performance.

Robust recall and balanced F1-score.

Computationally more intensive.

📈 Visualizations
ROC Curves for both models plotted and compared.

Confusion Matrices computed and interpreted.

Tree diagram of one decision tree in the ensemble displayed using view().

🔍 Key Takeaways
DTE outperforms NB across nearly all evaluation metrics.

SMOTE proved essential in handling class imbalance and improving model fairness.

NB remains a solid baseline with minimal computational cost.

Ensemble methods offer superior predictive power at the cost of complexity.

💡 Lessons & Future Work
Apply cross-validation for better generalization.

Explore Random Forest or Gradient Boosting as alternatives.

Optimize Naïve Bayes using kernel density estimation.

Enhance recall of NB with feature selection or transformation.

Incorporate model updating techniques to address scalability in real-time applications.
