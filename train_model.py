import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv(r"C:\Users\LENOVO\Downloads\archive (13)\train_u6lujuX_CVtuZ9i.csv")

print("Dataset loaded successfully")
print("Original shape:", df.shape)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

# Total income
df["TotalIncome"] = (
    df["ApplicantIncome"] +
    df["CoapplicantIncome"]
)

# EMI
df["EMI"] = (
    df["LoanAmount"] /
    df["Loan_Amount_Term"]
)

# Approximate balance income
df["BalanceIncome"] = (
    df["TotalIncome"] -
    (df["EMI"] * 1000)
)

# Income to loan ratio
df["IncomeLoanRatio"] = (
    df["TotalIncome"] /
    (df["LoanAmount"] + 1)
)

# Applicant income group
df["IncomeGroup"] = pd.cut(
    df["TotalIncome"],
    bins=[0, 2500, 5000, 10000, float("inf")],
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ],
    include_lowest=True
)

# Loan amount group
df["LoanGroup"] = pd.cut(
    df["LoanAmount"],
    bins=[0, 100, 200, 400, float("inf")],
    labels=[
        "Small",
        "Medium",
        "Large",
        "Very Large"
    ],
    include_lowest=True
)


# ============================================================
# 3. DROP UNNECESSARY COLUMN
# ============================================================

df.drop(
    "Loan_ID",
    axis=1,
    inplace=True
)


# ============================================================
# 4. FEATURES AND TARGET
# ============================================================

X = df.drop(
    "Loan_Status",
    axis=1
)

y = df["Loan_Status"].map({
    "N": 0,
    "Y": 1
})


# ============================================================
# 5. IDENTIFY COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_columns)

print("\nNumeric columns:")
print(numeric_columns)


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 8. MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ============================================================
# 9. TRAIN AND COMPARE MODELS
# ============================================================

results = {}

best_model = None
best_model_name = None
best_accuracy = 0


for name, classifier in models.items():

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results[name] = accuracy

    print(
        f"{name}: "
        f"{accuracy * 100:.2f}%"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = pipeline
        best_model_name = name


# ============================================================
# 10. BEST MODEL
# ============================================================

print("\n===================================")
print("BEST MODEL")
print("===================================")

print("Model:", best_model_name)
print(
    "Accuracy:",
    f"{best_accuracy * 100:.2f}%"
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

best_predictions = best_model.predict(
    X_test
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        best_predictions
    )
)


# ============================================================
# 12. SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "loan_model.pkl"
)

joblib.dump(
    {
        "model_name": best_model_name,
        "accuracy": best_accuracy,
        "results": results
    },
    "model_info.pkl"
)


print("\nModel saved successfully!")
print("Created:")
print("loan_model.pkl")
print("model_info.pkl")