import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="LoanWise AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# LOAD DATA / MODEL
# ============================================================
@st.cache_data
def load_data():
    return pd.read_csv(r"train_u6lujuX_CVtuZ9i.csv")


@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")


@st.cache_resource
def load_model_info():
    return joblib.load("model_info.pkl")


df = load_data()
model = load_model()
model_info = load_model_info()


# ============================================================
# SHAP BACKGROUND DATA
# ============================================================
@st.cache_data
def create_shap_background(data):
    background = data.copy()

    # Missing-value handling for the explanation background.
    categorical_cols = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
    ]

    numeric_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]

    for col in categorical_cols:
        if col in background.columns:
            background[col] = background[col].fillna(
                background[col].mode()[0]
            )

    for col in numeric_cols:
        if col in background.columns:
            background[col] = background[col].fillna(
                background[col].median()
            )

    # Feature engineering must match the model input.
    background["TotalIncome"] = (
        background["ApplicantIncome"]
        + background["CoapplicantIncome"]
    )

    background["IncomeGroup"] = pd.cut(
        background["TotalIncome"],
        bins=[0, 2500, 5000, 10000, float("inf")],
        labels=["Low", "Medium", "High", "Very High"],
        include_lowest=True,
    ).astype(str)

    background["LoanGroup"] = pd.cut(
        background["LoanAmount"],
        bins=[0, 100, 200, 400, float("inf")],
        labels=["Small", "Medium", "Large", "Very Large"],
        include_lowest=True,
    ).astype(str)

    background["EMI"] = (
        background["LoanAmount"]
        / background["Loan_Amount_Term"]
    )

    background["BalanceIncome"] = (
        background["TotalIncome"]
        - (background["EMI"] * 1000)
    )

    background["IncomeLoanRatio"] = (
        background["TotalIncome"]
        / (background["LoanAmount"] + 1)
    )

    feature_columns = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "Property_Area",
        "TotalIncome",
        "EMI",
        "BalanceIncome",
        "IncomeLoanRatio",
        "IncomeGroup",
        "LoanGroup",
    ]

    return background[feature_columns]


shap_background = create_shap_background(df)
shap_background = shap_background.sample(
    n=min(40, len(shap_background)),
    random_state=42,
)


@st.cache_resource
def create_shap_explainer(background):
    feature_names = background.columns.tolist()

    def predict_approval(data):
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=feature_names)
        return model.predict_proba(data)[:, 1]

    return shap.Explainer(
        predict_approval,
        background,
        algorithm="permutation",
    )


# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown(
    """
    <style>
    .stApp { background-color: #f5f7fb; }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #172554;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #172554;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .card {
        padding: 25px;
        border-radius: 18px;
        background-color: white;
        box-shadow: 0 4px 18px rgba(0,0,0,0.07);
        margin-bottom: 20px;
    }

    .feature-card {
        padding: 25px;
        border-radius: 18px;
        background-color: white;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        min-height: 180px;
    }

    .feature-icon { font-size: 35px; }

    .feature-title {
        font-size: 20px;
        font-weight: 700;
        color: #172554;
    }

    .feature-text {
        color: #64748b;
        font-size: 15px;
    }

    section[data-testid="stSidebar"] {
        background-color: #172554;
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 17px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("# 🏦 LoanWise AI")
    st.markdown("### Smart Loan Assessment")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔮 Loan Prediction",
            "📊 EDA Dashboard",
            "🤖 Model Performance",
            "ℹ️ About Project",
        ],
    )

    st.markdown("---")
    st.markdown("### 🤖 Current Model")
    st.write(model_info["model_name"])

    st.markdown("### 📈 Test Accuracy")
    st.write(f"{model_info['accuracy'] * 100:.2f}%")

    st.markdown("---")
    st.caption("LoanWise AI • Machine Learning Project")


# ============================================================
# HOME
# ============================================================
if page == "🏠 Home":
    st.markdown(
        '<div class="main-title">🏦 LoanWise AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">AI-powered Loan Approval Prediction System</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
        <h2 style="color:#172554;">Welcome to LoanWise AI 👋</h2>
        <p style="font-size:17px;color:#64748b;">
        LoanWise AI analyzes applicant personal, financial, credit and
        property information to estimate the likelihood of loan approval.
        The application also uses SHAP explainability to show which
        features influenced an individual prediction.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Records", len(df))
    with col2:
        st.metric("Features", 17)
    with col3:
        st.metric("ML Model", model_info["model_name"])
    with col4:
        st.metric("Accuracy", f"{model_info['accuracy'] * 100:.2f}%")

    st.markdown(
        '<div class="section-title">✨ Project Features</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
            <div class="feature-icon">🔮</div>
            <div class="feature-title">Loan Prediction</div>
            <p class="feature-text">
            Enter applicant details and receive an approval probability.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">EDA Dashboard</div>
            <p class="feature-text">
            Explore loan approval, credit, income and property patterns.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">SHAP Explainability</div>
            <p class="feature-text">
            Understand which features push an individual prediction
            toward or away from approval.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">🚀 How It Works</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
        <h3>1️⃣ Enter Applicant Details</h3>
        <p>Provide personal, financial, credit and property information.</p>
        <h3>2️⃣ Data Processing</h3>
        <p>The input is transformed to match the training features.</p>
        <h3>3️⃣ Machine Learning Prediction</h3>
        <p>The trained model calculates approval and rejection probabilities.</p>
        <h3>4️⃣ SHAP Explanation</h3>
        <p>SHAP identifies the strongest factors influencing the prediction.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAN PREDICTION
# ============================================================
elif page == "🔮 Loan Prediction":
    st.markdown(
        '<div class="main-title">🔮 Loan Prediction</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Enter applicant information to predict loan approval</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">👤 Applicant Information</div>',
        unsafe_allow_html=True,
    )

    applicant_name = st.text_input(
        "Applicant Name",
        placeholder="Enter applicant full name",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        married = st.selectbox("Married", ["Yes", "No"])
    with col3:
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

    col1, col2 = st.columns(2)
    with col1:
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with col2:
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    st.markdown(
        '<div class="section-title">💰 Financial Information</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        applicant_income = st.number_input(
            "Applicant Income (₹)",
            min_value=0,
            value=50000,
            step=500,
        )
    with col2:
        coapplicant_income = st.number_input(
            "Coapplicant Income (₹)",
            min_value=0,
            value=0,
            step=500,
        )
    with col3:
        loan_amount = st.number_input(
            "Loan Amount (₹)",
            min_value=500,
            value=100000,
            step=500,
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        loan_term = st.selectbox(
            "Loan Term (months)",
            [60, 120, 180, 240, 300, 360, 480],
            index=5,
        )
    with col2:
        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda x: (
                "Good Credit History" if x == 1.0 else "Poor Credit History"
            ),
        )
    with col3:
        property_area = st.selectbox(
            "Property Area",
            ["Urban", "Semiurban", "Rural"],
        )

    # --------------------------------------------------------
    # IMPORTANT UNIT CONVERSION
    # --------------------------------------------------------
    # ApplicantIncome and CoapplicantIncome stay in their
    # original dataset scale. LoanAmount is stored in thousands
    # in the Kaggle dataset, while the UI displays rupees.
    loan_amount_model = loan_amount / 1000.0

    total_income = applicant_income + coapplicant_income

    emi = loan_amount_model / loan_term
    balance_income = total_income - (emi * 1000)
    income_loan_ratio = total_income / (loan_amount_model + 1)

    st.markdown(
        '<div class="section-title">📊 Financial Summary</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Income", f"₹{total_income:,.0f}")
    with col2:
        st.metric("Estimated EMI", f"₹{emi * 1000:,.0f}")
    with col3:
        st.metric("Balance Income", f"₹{balance_income:,.0f}")

    st.caption(
        "Loan amount is displayed in rupees but converted to the dataset's "
        "thousand-unit scale before prediction."
    )

    st.markdown("---")

    predict = st.button(
        "🔮 Predict Loan Approval",
        use_container_width=True,
    )

    if predict:
        if not applicant_name.strip():
            st.warning("⚠️ Please enter the applicant name.")
            st.stop()

        # Feature engineering must match training.
        if total_income <= 2500:
            income_group = "Low"
        elif total_income <= 5000:
            income_group = "Medium"
        elif total_income <= 10000:
            income_group = "High"
        else:
            income_group = "Very High"

        if loan_amount_model <= 100:
            loan_group = "Small"
        elif loan_amount_model <= 200:
            loan_group = "Medium"
        elif loan_amount_model <= 400:
            loan_group = "Large"
        else:
            loan_group = "Very Large"

        input_data = pd.DataFrame(
            {
                "Gender": [gender],
                "Married": [married],
                "Dependents": [dependents],
                "Education": [education],
                "Self_Employed": [self_employed],
                "ApplicantIncome": [applicant_income],
                "CoapplicantIncome": [coapplicant_income],
                "LoanAmount": [loan_amount_model],
                "Loan_Amount_Term": [loan_term],
                "Credit_History": [credit_history],
                "Property_Area": [property_area],
                "TotalIncome": [total_income],
                "EMI": [emi],
                "BalanceIncome": [balance_income],
                "IncomeLoanRatio": [income_loan_ratio],
                "IncomeGroup": [income_group],
                "LoanGroup": [loan_group],
            }
        )

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]
        except Exception as error:
            st.error("❌ The model could not process this input.")
            st.exception(error)
            st.stop()

        # Make probability mapping robust to either [0, 1] or ['N', 'Y'].
        classes = list(getattr(model, "classes_", [0, 1]))

        try:
            approval_index = classes.index(1)
            rejection_index = classes.index(0)
        except ValueError:
            try:
                approval_index = classes.index("Y")
                rejection_index = classes.index("N")
            except ValueError:
                approval_index = 1
                rejection_index = 0

        approval_probability = float(probability[approval_index])
        rejection_probability = float(probability[rejection_index])

        st.markdown("---")
        st.markdown(
            '<div class="section-title">🎯 Prediction Result</div>',
            unsafe_allow_html=True,
        )

        approved_prediction = prediction in [1, "Y", "Yes", True]

        if approved_prediction:
            st.success(f"## ✅ Loan Approved for {applicant_name}")
            st.info(
                "The model predicts a higher likelihood of approval based on "
                "the applicant information provided."
            )
        else:
            st.error(f"## ❌ Loan Rejected for {applicant_name}")

            reasons = []
            if credit_history == 0.0:
                reasons.append("poor credit history")
            if total_income < 3000:
                reasons.append("relatively low combined income")
            if loan_amount_model > total_income * 2:
                reasons.append("a high loan amount compared with income")
            if balance_income < 0:
                reasons.append("a high estimated repayment burden")

            if reasons:
                if len(reasons) == 1:
                    reason_text = reasons[0]
                elif len(reasons) == 2:
                    reason_text = f"{reasons[0]} and {reasons[1]}"
                else:
                    reason_text = ", ".join(reasons[:-1]) + f", and {reasons[-1]}"

                st.info(
                    f"💡 **Possible contributing factors:** {reason_text}. "
                    "These are indicative factors based on the applicant's "
                    "information, not the exact reason produced by the model."
                )
            else:
                st.info(
                    "💡 **Why might the loan be rejected?** The applicant's "
                    "overall profile has a lower predicted probability of "
                    "approval compared with the patterns learned from the "
                    "training data."
                )

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Approval Probability",
                f"{approval_probability * 100:.2f}%",
            )
        with col2:
            st.metric(
                "Rejection Probability",
                f"{rejection_probability * 100:.2f}%",
            )

        st.write("Approval Probability")
        st.progress(approval_probability)

        # ====================================================
        # SHAP EXPLAINABILITY
        # ====================================================
        st.markdown("---")
        st.markdown(
            '<div class="section-title">🧠 Why did the model make this prediction?</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "SHAP explains this individual prediction by estimating how each "
            "feature changes the model's approval probability relative to the "
            "background applicants."
        )

        with st.spinner("Calculating SHAP explanation..."):
            try:
                explainer = create_shap_explainer(shap_background)
                shap_explanation = explainer(input_data, max_evals=100)

                values = np.asarray(shap_explanation.values)
                if values.ndim == 3:
                    # Some SHAP/model combinations return an output dimension.
                    values = values[0, :, 0]
                elif values.ndim == 2:
                    values = values[0]
                else:
                    values = values.reshape(-1)

                feature_names = input_data.columns.tolist()

                shap_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Input Value": [input_data.iloc[0][c] for c in feature_names],
                        "SHAP Value": values,
                    }
                )
                shap_df["Absolute Impact"] = shap_df["SHAP Value"].abs()
                shap_df = shap_df.sort_values(
                    "Absolute Impact", ascending=False
                )

                top_features = shap_df.head(8).copy()
                plot_data = top_features.sort_values("SHAP Value")

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.barh(
                    plot_data["Feature"],
                    plot_data["SHAP Value"],
                )
                ax.axvline(0, linewidth=1)
                ax.set_xlabel("SHAP Value")
                ax.set_ylabel("Feature")
                ax.set_title("Top Feature Contributions to This Prediction")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                col1, col2 = st.columns(2)

                positive = shap_df[shap_df["SHAP Value"] > 0].head(5)
                negative = shap_df[shap_df["SHAP Value"] < 0].head(5)

                with col1:
                    st.subheader("🟢 Factors Supporting Approval")
                    if positive.empty:
                        st.write("No positive SHAP contributions were found.")
                    else:
                        for _, row in positive.iterrows():
                            st.success(
                                f"**{row['Feature']}** — SHAP {row['SHAP Value']:+.4f}"
                            )

                with col2:
                    st.subheader("🔴 Factors Reducing Approval")
                    if negative.empty:
                        st.write("No negative SHAP contributions were found.")
                    else:
                        for _, row in negative.iterrows():
                            st.error(
                                f"**{row['Feature']}** — SHAP {row['SHAP Value']:+.4f}"
                            )

                st.subheader("💡 Simple Explanation")

                strongest_positive = positive.iloc[0] if not positive.empty else None
                strongest_negative = negative.iloc[0] if not negative.empty else None

                if approved_prediction:
                    if strongest_positive is not None:
                        st.success(
                            f"The strongest factor supporting approval was "
                            f"**{strongest_positive['Feature']}**."
                        )
                    if strongest_negative is not None:
                        st.warning(
                            f"**{strongest_negative['Feature']}** pushed the prediction "
                            "in the opposite direction."
                        )
                else:
                    if strongest_negative is not None:
                        st.warning(
                            f"The strongest factor reducing approval was "
                            f"**{strongest_negative['Feature']}**."
                        )
                    if strongest_positive is not None:
                        st.info(
                            f"**{strongest_positive['Feature']}** provided some support "
                            "toward approval."
                        )

                with st.expander("🔍 View Detailed SHAP Values"):
                    display_shap = shap_df[
                        ["Feature", "Input Value", "SHAP Value"]
                    ].copy()
                    display_shap["Impact"] = display_shap["SHAP Value"].apply(
                        lambda x: "Supports Approval" if x > 0 else "Reduces Approval"
                    )
                    st.dataframe(
                        display_shap,
                        use_container_width=True,
                        hide_index=True,
                    )

            except Exception as error:
                st.warning(
                    "SHAP explanation could not be generated for this model. "
                    "The prediction itself is still valid."
                )
                st.caption(f"SHAP technical details: {error}")

        # ====================================================
        # APPLICATION SUMMARY
        # ====================================================
        st.markdown("---")
        st.markdown(
            '<div class="section-title">📋 Application Summary</div>',
            unsafe_allow_html=True,
        )

        summary = pd.DataFrame(
            {
                "Field": [
                    "Applicant Name",
                    "Gender",
                    "Married",
                    "Dependents",
                    "Education",
                    "Self Employed",
                    "Applicant Income",
                    "Coapplicant Income",
                    "Total Income",
                    "Loan Amount",
                    "Loan Term",
                    "Credit History",
                    "Property Area",
                ],
                "Value": [
                    applicant_name,
                    gender,
                    married,
                    dependents,
                    education,
                    self_employed,
                    f"₹{applicant_income:,.0f}",
                    f"₹{coapplicant_income:,.0f}",
                    f"₹{total_income:,.0f}",
                    f"₹{loan_amount:,.0f}",
                    f"{loan_term} months",
                    "Good" if credit_history == 1.0 else "Poor",
                    property_area,
                ],
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            '<div class="section-title">💡 Assessment</div>',
            unsafe_allow_html=True,
        )

        if approval_probability >= 0.75:
            st.success("The model shows a strong likelihood of loan approval.")
        elif approval_probability >= 0.50:
            st.info("The model shows a moderate likelihood of loan approval.")
        else:
            st.warning("The model shows a lower likelihood of loan approval.")


# ============================================================
# EDA DASHBOARD
# ============================================================
elif page == "📊 EDA Dashboard":
    st.markdown(
        '<div class="main-title">📊 EDA Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Explore patterns and relationships in the loan dataset</div>',
        unsafe_allow_html=True,
    )

    approved = (df["Loan_Status"] == "Y").sum()
    rejected = (df["Loan_Status"] == "N").sum()
    approval_rate = approved / len(df) * 100

    st.markdown(
        '<div class="section-title">📌 Dataset Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Applications", len(df))
    with col2:
        st.metric("Approved", approved)
    with col3:
        st.metric("Rejected", rejected)
    with col4:
        st.metric("Approval Rate", f"{approval_rate:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loan Approval Distribution")
        status_counts = df["Loan_Status"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(
            ["Approved", "Rejected"],
            [status_counts.get("Y", 0), status_counts.get("N", 0)],
        )
        ax.set_ylabel("Number of Applications")
        ax.set_title("Loan Approval Distribution")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Credit History vs Loan Status")
        credit_table = pd.crosstab(df["Credit_History"], df["Loan_Status"])
        st.bar_chart(credit_table)

    st.markdown("---")
    st.subheader("💰 Income Analysis")
    st.line_chart(
        df[["ApplicantIncome", "CoapplicantIncome"]].head(50)
    )

    st.subheader("🏦 Loan Amount Distribution")
    loan_data = df["LoanAmount"].dropna()
    fig, ax = plt.subplots()
    ax.hist(loan_data, bins=20)
    ax.set_xlabel("Loan Amount (dataset units / ₹ thousands)")
    ax.set_ylabel("Number of Applicants")
    ax.set_title("Loan Amount Distribution")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("🏠 Property Area Analysis")
    st.bar_chart(df["Property_Area"].value_counts())


# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "🤖 Model Performance":
    st.markdown(
        '<div class="main-title">🤖 Model Performance</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Machine learning model evaluation</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", model_info["model_name"])
    with col2:
        st.metric("Test Accuracy", f"{model_info['accuracy'] * 100:.2f}%")
    with col3:
        st.metric("Training Records", int(len(df) * 0.8))

    st.markdown("---")
    st.subheader("📈 Model Comparison")

    results = model_info.get("results", {})
    if results:
        comparison = pd.DataFrame(
            {
                "Model": list(results.keys()),
                "Accuracy": [value * 100 for value in results.values()],
            }
        )
        st.bar_chart(comparison.set_index("Model"))
        st.dataframe(
            comparison.style.format({"Accuracy": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Model comparison results are not available in model_info.pkl.")

    st.markdown("---")
    st.subheader("🧠 Machine Learning Pipeline")
    st.markdown(
        """
        <div class="card">
        <h3>1️⃣ Data Collection</h3>
        <p>Loan applicant information is loaded from the Kaggle loan dataset.</p>
        <h3>2️⃣ Data Preprocessing</h3>
        <p>Missing values are handled using appropriate imputation.</p>
        <h3>3️⃣ Feature Engineering</h3>
        <p>Total income, income group, loan group, EMI, balance income and income-to-loan ratio are generated.</p>
        <h3>4️⃣ Encoding</h3>
        <p>Categorical variables are transformed using the preprocessing used by the saved model.</p>
        <h3>5️⃣ Model Training</h3>
        <p>Classification models are compared and the best-performing saved model is used.</p>
        <h3>6️⃣ Explainability</h3>
        <p>SHAP is used to explain individual predictions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ABOUT PROJECT
# ============================================================
elif page == "ℹ️ About Project":
    st.markdown(
        '<div class="main-title">ℹ️ About Project</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">LoanWise AI — Machine Learning Loan Assessment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
        <h2>🎯 Project Objective</h2>
        <p>
        The objective of LoanWise AI is to predict whether a loan application
        is likely to be approved based on applicant personal, financial,
        credit and property information.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🛠️ Technologies Used")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.info("🐍 Python")
    with col2:
        st.info("🐼 Pandas")
    with col3:
        st.info("🤖 Scikit-learn")
    with col4:
        st.info("🌐 Streamlit")
    with col5:
        st.info("🧠 SHAP")

    st.subheader("📚 Machine Learning Techniques")
    st.markdown(
        """
        - Missing value handling
        - Exploratory Data Analysis
        - Feature engineering
        - Categorical encoding
        - Model comparison
        - Classification
        - Probability prediction
        - SHAP explainability
        - Streamlit deployment
        """
    )

    st.subheader("📊 Input Features")
    features = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self Employed",
        "Applicant Income",
        "Coapplicant Income",
        "Loan Amount",
        "Loan Term",
        "Credit History",
        "Property Area",
    ]
    st.dataframe(
        pd.DataFrame({"Feature": features}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("🧠 What is SHAP?")
    st.markdown(
        """
        **SHAP (SHapley Additive exPlanations)** explains an individual
        machine-learning prediction by estimating how each feature changes
        the model output relative to a background set of applicants.

        - **Positive SHAP value:** pushes the approval probability upward.
        - **Negative SHAP value:** pushes the approval probability downward.

        SHAP makes the prediction easier to interpret, but it should not be
        treated as a banking decision or as proof that one feature alone
        caused approval or rejection.
        """
    )

    st.subheader("⚠️ Disclaimer")
    st.warning(
        "This application is developed for educational and demonstration "
        "purposes. The prediction should not be used as an actual banking "
        "or financial decision."
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "🏦 LoanWise AI | Python • Pandas • Scikit-learn • Streamlit • SHAP"
)