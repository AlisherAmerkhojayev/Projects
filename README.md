# Predicting Customer Churn Using Decision Tree Ensembles & Naïve Bayes

---

##  Personal Motivation

This project was designed as a comprehensive, end-to-end application of supervised machine learning for a business-critical problem: **predicting customer churn**. I undertook this work to deepen my experience in:

- **Data analytics pipelines**, from preprocessing to evaluation
- Handling **imbalanced datasets**, common in finance and marketing
- Comparing **model interpretability vs. predictive performance**
- Gaining practical experience with **model optimization, diagnostics, and validation**

As someone targeting roles in **data analytics and financial services**, I wanted to demonstrate:
- My understanding of **customer behavior modeling**
- Proficiency in **rigorous model evaluation**
- Comfort with **translating technical insights into business impact**

---

## 📌 Project Overview

This project applies two well-known classification algorithms—**Naïve Bayes** and **Decision Tree Ensembles**—to predict which customers are likely to **churn (leave)** a retail bank. The business relevance of this task is significant: retaining existing customers is typically far more cost-effective than acquiring new ones.

However, churn data presents **technical challenges**:
- **Highly imbalanced** target classes
- Need to balance **speed vs. accuracy**
- Importance of **interpretable** models in high-stakes decisions

This project investigates these trade-offs through:
- **SMOTE oversampling** to address class imbalance
- **Hyperparameter tuning** for model refinement
- **Robust evaluation metrics** to guide model selection

---

## 🎯 Goals & Questions Addressed

- Can a simple model like **Naïve Bayes** provide actionable insight into customer churn?
- Does an ensemble of trees significantly outperform simpler approaches—and is it worth the extra complexity?
- How can we best handle **class imbalance**, which skews predictions toward the majority class?
- What metrics best reflect the **real-world utility** of our model—**accuracy**, **precision**, or **recall**?

---

## 🗃️ Dataset Description

- **Name**: Churn_Modelling.csv  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/)  
- **Observations**: 10,000 customer records  
- **Target Variable**: `Exited` (1 = churned, 0 = retained)  
- **Features Used**: 10 (including `CreditScore`, `Age`, `Balance`, `EstimatedSalary`, etc.)

Notably:
- Categorical features (`Geography`, `Gender`) were **converted to numeric format** using `grp2idx`. This enables the use of ML algorithms that require numerical inputs.
- The dataset was **highly imbalanced** (~20% churn rate), which mirrors real-world financial data and requires careful handling to avoid biased models.

---

## 🧠 Why SMOTE?

In most financial datasets, **non-churned customers dominate**. This leads models to **optimize for accuracy by predicting the majority class**, which results in poor performance for the class we actually care about (churned customers).

To mitigate this, I used **SMOTE** (Synthetic Minority Over-sampling Technique) to:
- **Balance the class distribution** in the training data
- Improve **recall** and **F1-score**, especially for the minority class
- Prevent overfitting by generating realistic synthetic samples, rather than duplicating data

---

## ⚙️ Modeling Approach

### **1. Naïve Bayes (Baseline Model)**

Chosen because:
- It’s **computationally efficient**
- Easy to **interpret and implement**
- Performs surprisingly well on **well-structured tabular data**

Trained using `fitcnb` with **normal (Gaussian)** distribution assumptions—appropriate given the **partial Gaussian behavior** of some features like balance and age.

---

### **2. Decision Tree Ensembles (Bagging)**

Selected for:
- Ability to **model nonlinear relationships**
- Naturally **handles interactions** between variables
- Robust to outliers and **less sensitive to scaling**

Used MATLAB’s `fitcensemble` with `'Method' = 'Bag'`, indicating **bootstrap aggregation**.

To fine-tune the model, I performed **grid search hyperparameter tuning** over:
- `MaxNumSplits`: controls tree depth
- `MinLeafSize`: prevents overfitting by requiring minimum observations in leaves
- `MinParentSize`: governs minimum samples before a split

Performance was validated using **AUC on a hold-out validation set**, aligning with business goals to minimize **false negatives** (i.e., missed churners).

---

## 📈 Performance Summary

| Metric        | Naïve Bayes | Decision Tree Ensemble |
|---------------|-------------|-------------------------|
| **Accuracy**  | 57.2%       | **76.6%**               |
| **Precision** | **0.93**    | 0.90                    |
| **Recall**    | 0.51        | **0.79**                |
| **F1-score**  | 0.636       | **0.84**                |
| **AUC**       | 0.78        | **0.81**                |

### Interpretation:
- **Naïve Bayes** was fast and delivered **high precision**, but suffered from **low recall**, meaning many churners were missed.
- **DTE** had strong **recall and F1-score**, indicating it captured more true churners and balanced performance.
- **AUC** further confirmed DTE’s superiority in ranking churn risk.

---

## 📊 Visual Outputs

- **ROC Curve**: Plotted for both models; DTE curve dominated NB’s.
- **Confusion Matrix**: Used to interpret precision/recall trade-offs.
- **Tree Visualization**: One trained decision tree was rendered using `view()` to enhance transparency.

---

## 🔍 Key Learnings

- **Class imbalance must be addressed** or results become misleading.
- **DTE is powerful but costly**—a practical insight for real-time or low-resource applications.
- **Naïve Bayes serves well as a quick diagnostic tool** or fallback model.
- **Evaluation metrics matter**—accuracy alone is insufficient for imbalanced classification.

---

## 🚀 Future Enhancements

- Integrate **cross-validation** to generalize results.
- Explore **Random Forests** or **Gradient Boosting** for higher performance.
- Enhance Naïve Bayes via **kernel density estimation**.
- Add **cost-sensitive learning** for better business alignment.
- Use **feature engineering** to improve interpretability and reduce dimensionality.
