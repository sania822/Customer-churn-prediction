#  Customer Churn Prediction

##  Overview

This project focuses on predicting customer churn using machine learning techniques. The goal is to identify customers who are likely to leave, enabling businesses to take proactive retention strategies.

---

##  Objectives

* Analyze customer behavior and churn patterns
* Build a classification model to predict churn
* Improve model performance using hyperparameter tuning
* Translate results into actionable business insights

---

## Dataset

The dataset includes customer-related features such as:

* Demographics
* Account information
* Service usage patterns
* Target variable: **Churn (0 = No, 1 = Yes)**

---

##  Exploratory Data Analysis (EDA)

* Identified class imbalance in churn data
* Observed key patterns between customer usage and churn
* Analyzed feature relationships influencing churn

---

##  Data Preprocessing & Feature Engineering

* Handled missing values
* Encoded categorical variables
* Scaled numerical features where required
* Prepared dataset for classification models

---

##  Model Building

### Models Implemented:

* Logistic Regression
* Random Forest Classifier
* Random Forest with **GridSearchCV (Hyperparameter Tuning)** 

---

##  Model Performance

###  Logistic Regression

* Accuracy: **75%**
* Recall (Churn class): **67%**
* F1 Score: **0.44**

Good baseline model but struggled with balanced performance.

---

###  Random Forest

* Accuracy: **94%**
* Precision (Churn class): **98%**
* Recall (Churn class): **58%**
* F1 Score: **0.73**

 High accuracy but lower recall for churn customers.

---

### 🔹 Tuned Random Forest (GridSearchCV) 

* Accuracy: **96%**
* Precision (Churn class): **97%**
* Recall (Churn class): **73%**
* F1 Score: **0.83**

 Best-performing model with balanced precision and recall.

---

##  Key Insights

* Class imbalance significantly affects churn prediction
* Recall is critical for identifying churn customers
* Hyperparameter tuning improved model generalization

---

##  Business Impact

* Helps identify customers at risk of leaving
* Enables targeted retention campaigns
* Improves customer lifetime value

---

##  Deployment (if applicable)

The model can be deployed using a web application to provide real-time churn predictions.

---

##  Project Structure

```id="9xgq4m"
churn_project/
│── notebook.ipynb
│── model.pkl
│── app.py (optional)
│── README.md
```

---

##  Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

##  Key Learnings

* Importance of handling imbalanced datasets
* Trade-off between precision and recall
* Role of hyperparameter tuning in improving performance
* Evaluating models beyond accuracy

---

##  Author

Sania

---

## ⭐ Conclusion

This project demonstrates how machine learning can be effectively used to predict customer churn and support data-driven retention strategies, with improved performance achieved through model tuning.
