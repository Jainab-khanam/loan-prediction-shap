# Loan Prediction Using Logistic Regression

A data science project to predict whether a customer will accept a **personal loan** based on demographic and financial information.

##  Dataset
The dataset includes features such as:
- Age, Experience, Income, Family, Education
- Credit card spending (CCAvg), Mortgage amount
- Online banking, CreditCard, CD/Investment account indicators

Source: [Kaggle Loan Dataset]

##  Exploratory Data Analysis (EDA)
- Investigated missing values, distributions, and feature correlations.
- Visualized loan approval rates by income, education, and account types.


##  Model Performance

Used `LogisticRegression` with preprocessing:

- **Pipeline**: SimpleImputer → StandardScaler → LogisticRegression
- **max_iter**: 1000 (to ensure convergence)

### 🔢 Results:
| Metric        | Class 0 (No Loan) | Class 1 (Loan Accepted) |
|---------------|------------------|--------------------------|
| Precision     | 0.96             | 0.85                     |
| Recall        | 0.99             | 0.66                     |
| F1-Score      | 0.97             | 0.74                     |

- **Overall Accuracy**: **95%**
- **Macro Avg F1-Score**: 0.86
- **Weighted Avg F1-Score**: 0.95

✅ This model balances accuracy and recall well, despite mild class imbalance.


##  Feature Importance (SHAP)
Used SHAP to identify top features influencing model predictions:

![SHAP Importance](images/shap_feature_importance.png)

- `Income` and `Education` are the most influential.
- Online, Age, and Experience have minor contributions.

##  Tech Stack
- Python (Pandas, NumPy, scikit-learn, SHAP, Matplotlib)
- Jupyter Notebook
- GitHub for version control and portfolio showcase

##  Future Improvements
- Apply SMOTE or class-weight adjustments for better recall.
- Try ensemble methods like XGBoost.
- Deploy using Streamlit.

---