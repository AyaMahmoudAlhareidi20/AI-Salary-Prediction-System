import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score


# ----------------------------------------
# 1) Load Data
# ----------------------------------------

data = pd.read_csv("global_ai_ml_data_salaries.csv")
data.drop_duplicates(inplace=True)

# Create binary salary target
median_salary = data["salary_in_usd"].median()
data["salary_level"] = (data["salary_in_usd"] > median_salary).astype(int)

# Main feature set
X = data.drop(columns=["salary_in_usd", "salary_level"])
y_reg = data["salary_in_usd"]       # For Regression
y_clf = data["salary_level"]        # For Classification


# ----------------------------------------
# 2) Preprocessing
# ----------------------------------------

numeric_cols = ["salary", "remote_ratio"]

ordinal_cols = ["company_size", "employment_type"]
ordinal_categories = [
    ["S", "M", "L"],
    ["CT", "FT", "PT", "FL"]
]

onehot_cols = ["job_title", "salary_currency", "employee_residence", "company_location"]

if "experience_level" in X.columns:
    onehot_cols.append("experience_level")

ordinal = OrdinalEncoder(categories=ordinal_categories)
onehot = OneHotEncoder(handle_unknown="ignore")
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", scaler, numeric_cols),
        ("ord", ordinal, ordinal_cols),
        ("ohe", onehot, onehot_cols)
    ]
)


# ----------------------------------------
# 3) Split for Regression + Classification
# ----------------------------------------

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)


# ----------------------------------------
# 4) Train Regression Models
# ----------------------------------------

models_reg = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9
    )
}

reg_results = {}

for name, model in models_reg.items():
    pipe = Pipeline([
        ("pre", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train_reg, y_train_reg)
    preds = pipe.predict(X_test_reg)
    r2 = r2_score(y_test_reg, preds)
    reg_results[name] = (r2, pipe)

# Pick Best Regression Model
best_reg_name = max(reg_results, key=lambda k: reg_results[k][0])
best_reg_model = reg_results[best_reg_name][1]


# ----------------------------------------
# 5) Train Logistic Regression
# ----------------------------------------

clf_model = Pipeline([
    ("pre", preprocessor),
    ("model", LogisticRegression(max_iter=500))
])

clf_model.fit(X_train_clf, y_train_clf)
clf_acc = accuracy_score(y_test_clf, clf_model.predict(X_test_clf))


# ----------------------------------------
# 6) Save Best Models
# ----------------------------------------

joblib.dump(best_reg_model, "best_reg_model.pkl")
joblib.dump(clf_model, "best_clf_model.pkl")

print("\n===================================")
print("✔ Best Regression Model :", best_reg_name)
print("✔ Saved as best_reg_model.pkl")
print("✔ Classification Accuracy :", clf_acc)
print("✔ Saved as best_clf_model.pkl")
print("===================================\n")
