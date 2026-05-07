
# 💳 Credit Evaluation Model

Built an end-to-end supervised machine learning pipeline that predicts loan approval decisions based on applicant financial and personal data. Built with Logistic Regression, KNN, and Naive Bayes — with full EDA, feature engineering, and model evaluation.

---

## 🚀 What It Does

Takes applicant data as input and predicts whether a loan should be **approved or rejected**, using a complete ML pipeline from raw data to evaluated model.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML Models | scikit-learn |
| Feature Encoding | LabelEncoder, OneHotEncoder |
| Scaling | StandardScaler |

---

## 📁 Project Structure

```
Credit-Evaluation-Model/
├── credit_wise.py         # Full ML pipeline
├── loan_approval_data.csv # Dataset (add manually)
└── README.md
```

---

## 🔄 Pipeline Overview

### 1. Data Loading & Exploration
- Loaded loan approval dataset
- Inspected shape, types, null values, and statistics

### 2. Data Preprocessing
- Imputed missing numerical values using **mean strategy**
- Imputed missing categorical values using **most frequent strategy**
- Dropped `Applicant_ID` (irrelevant identifier)

### 3. Exploratory Data Analysis (EDA)
- Loan approval distribution (pie chart)
- Education level breakdown (bar chart)
- Income distributions (histograms)
- Outlier detection via box plots (Income, Credit Score, DTI Ratio, Savings)
- Credit score vs loan approval analysis
- Correlation heatmap across all features

### 4. Encoding
- `LabelEncoder` for Education Level and Loan Approved (target)
- `OneHotEncoder` for Employment Status, Marital Status, Loan Purpose, Property Area, Gender, Employer Category

### 5. Feature Engineering
- `DTI_Ratio_sq` — squared DTI ratio to capture non-linear relationships
- `Credit_Score_sq` — squared credit score
- `Applicant_Income_Log` — log-transformed income to reduce skew

### 6. Model Training & Evaluation

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear classifier |
| KNN | Tested with k = 5, 7, 9, 11 |
| Naive Bayes | Best model based on precision |

**Metrics used:** Accuracy · Precision · Recall · F1 Score · Confusion Matrix

### 7. Best Model
> ✅ **Naive Bayes** — selected based on highest precision score

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/credit-evaluation-model.git
cd credit-evaluation-model
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Add the dataset
Place `loan_approval_data.csv` in the root directory.

### 4. Run the pipeline
```bash
python credit_wise.py
```

---

## 📊 Features Used

| Feature | Type |
|---------|------|
| Applicant_Income | Numerical |
| Coapplicant_Income | Numerical |
| Credit_Score | Numerical |
| DTI_Ratio | Numerical |
| Savings | Numerical |
| Education_Level | Categorical |
| Employment_Status | Categorical |
| Marital_Status | Categorical |
| Loan_Purpose | Categorical |
| Property_Area | Categorical |
| Gender | Categorical |
| Employer_Category | Categorical |

**Target:** `Loan_Approved` (Binary — Yes/No)

---

## 🔮 Future Improvements

- [ ] Add Random Forest and XGBoost models
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Deploy as a web app using Streamlit
- [ ] Add SHAP values for model explainability

---

## 👤 Author

**Adarsh Verma**
- LinkedIn: [linkedin.com/in/adarshverma-84499b394](https://www.linkedin.com/in/adarshverma-84499b394)
- GitHub: [github.com/your-username](https://github.com/your-username)

---


