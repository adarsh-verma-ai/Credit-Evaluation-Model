import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(
    page_title="Credit Evaluation Model",
    page_icon="💳",
    layout="centered"
)

# ─── Train model on startup ───────────────────────────────────────────────────

@st.cache_resource
def train_model():
    df = pd.read_csv("loan_approval_data.csv")

    # Separate column types
    numerical_cols = df.select_dtypes(include=["number"]).columns
    categorical_cols_all = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                            "Property_Area", "Gender", "Employer_Category"]

    # Impute missing values
    num_imp = SimpleImputer(strategy="mean")
    df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

    cat_imp = SimpleImputer(strategy="most_frequent")
    df[categorical_cols_all] = cat_imp.fit_transform(df[categorical_cols_all])

    # Drop ID
    df = df.drop("Applicant_ID", axis=1)

    # Label encode
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = le.fit_transform(df["Loan_Approved"])

    # One-hot encode
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(df[categorical_cols_all])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(categorical_cols_all), index=df.index)
    df = pd.concat([df.drop(columns=categorical_cols_all), encoded_df], axis=1)

    # Feature engineering
    df["DTI_Ratio_sq"]          = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"]       = df["Credit_Score"] ** 2
    df["Applicant_Income_Log"]  = np.log1p(df["Applicant_Income"])

    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio", "Applicant_Income"])
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
        "Precision": round(precision_score(y_test, y_pred) * 100, 2),
        "Recall":    round(recall_score(y_test, y_pred) * 100, 2),
        "F1 Score":  round(f1_score(y_test, y_pred) * 100, 2),
    }

    return model, scaler, ohe, le, X.columns.tolist(), metrics


# ─── UI ───────────────────────────────────────────────────────────────────────

st.title("💳 Credit Evaluation Model")
st.caption("Predicts loan approval based on applicant financial and personal data.")

try:
    model, scaler, ohe, le, feature_cols, metrics = train_model()

    # Model metrics
    with st.expander("📊 Model Performance (Naive Bayes)", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{metrics['Accuracy']}%")
        col2.metric("Precision", f"{metrics['Precision']}%")
        col3.metric("Recall",    f"{metrics['Recall']}%")
        col4.metric("F1 Score",  f"{metrics['F1 Score']}%")

    st.divider()
    st.subheader("Enter Applicant Details")

    # ── Input form ──────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        applicant_income    = st.number_input("Applicant Income (₹)", min_value=0, value=50000, step=1000)
        coapplicant_income  = st.number_input("Co-applicant Income (₹)", min_value=0, value=0, step=1000)
        credit_score        = st.slider("Credit Score", min_value=300, max_value=900, value=700)
        dti_ratio           = st.number_input("DTI Ratio (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
        savings             = st.number_input("Savings (₹)", min_value=0, value=100000, step=5000)
        loan_amount         = st.number_input("Loan Amount (₹)", min_value=0, value=200000, step=5000)

    with col2:
        education_level     = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        employment_status   = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Unemployed"])
        marital_status      = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        loan_purpose        = st.selectbox("Loan Purpose", ["Home", "Education", "Business", "Personal"])
        property_area       = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
        gender              = st.selectbox("Gender", ["Male", "Female"])
        employer_category   = st.selectbox("Employer Category", ["Government", "Private", "Self"])

    st.divider()

    # ── Predict ─────────────────────────────────────────────────────────────
    if st.button("🔍 Evaluate Loan Application", use_container_width=True):

        # Build input row
        cat_input = pd.DataFrame([[
            employment_status, marital_status, loan_purpose,
            property_area, gender, employer_category
        ]], columns=["Employment_Status", "Marital_Status", "Loan_Purpose",
                     "Property_Area", "Gender", "Employer_Category"])

        encoded_input = ohe.transform(cat_input)
        encoded_input_df = pd.DataFrame(
            encoded_input,
            columns=ohe.get_feature_names_out(
                ["Employment_Status", "Marital_Status", "Loan_Purpose",
                 "Property_Area", "Gender", "Employer_Category"]
            )
        )

        edu_encoded = le.transform([education_level])[0]

        numerical_input = pd.DataFrame([[
            applicant_income, coapplicant_income, savings, loan_amount, edu_encoded
        ]], columns=["Applicant_Income", "Coapplicant_Income", "Savings", "Loan_Amount", "Education_Level"])

        # Feature engineering
        numerical_input["DTI_Ratio_sq"]         = dti_ratio ** 2
        numerical_input["Credit_Score_sq"]       = credit_score ** 2
        numerical_input["Applicant_Income_Log"]  = np.log1p(applicant_income)

        full_input = pd.concat([numerical_input, encoded_input_df], axis=1)

        # Align columns with training
        full_input = full_input.reindex(columns=feature_cols, fill_value=0)

        input_scaled = scaler.transform(full_input)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0]

        st.divider()

        if prediction == 1:
            st.success("✅ Loan Approved")
            st.progress(float(probability[1]), text=f"Approval confidence: {round(probability[1]*100, 1)}%")
        else:
            st.error("❌ Loan Not Approved")
            st.progress(float(probability[0]), text=f"Rejection confidence: {round(probability[0]*100, 1)}%")

        with st.expander("📋 Input Summary"):
            summary = {
                "Applicant Income": f"₹{applicant_income:,}",
                "Co-applicant Income": f"₹{coapplicant_income:,}",
                "Credit Score": credit_score,
                "DTI Ratio": f"{dti_ratio}%",
                "Savings": f"₹{savings:,}",
                "Loan Amount": f"₹{loan_amount:,}",
                "Education": education_level,
                "Employment": employment_status,
                "Marital Status": marital_status,
                "Loan Purpose": loan_purpose,
                "Property Area": property_area,
                "Gender": gender,
                "Employer Category": employer_category,
            }
            for key, val in summary.items():
                st.write(f"**{key}:** {val}")

except FileNotFoundError:
    st.error("⚠️ `loan_approval_data.csv` not found. Please add the dataset to the project directory and restart the app.")
