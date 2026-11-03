
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading modeling dataset...")

df = pd.read_csv("data/processed/modeling_data.csv")

print("Dataset shape:", df.shape)


# ==========================================
# 2. DEFINE FEATURES AND TARGET
# ==========================================

target = "log_valuation"

groups = df["object_id"]

numeric_features = [
    "raised_amount_usd",
    "company_age",
    "relationships"
]

categorical_features = [
    "funding_round_type",
    "funding_round_code",
    "category_code",
    "country_code"
]

X = df[numeric_features + categorical_features]
y = df[target]


# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print(
    "Unique training startups:",
    groups.iloc[train_idx].nunique()
)

print(
    "Unique testing startups:",
    groups.iloc[test_idx].nunique()
)


# ==========================================
# 4. PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {

    "Baseline": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DummyRegressor(strategy="median"))
        ]
    ),

    "Random Forest": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    ),

    "Gradient Boosting": Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                )
            )
        ]
    )
}


# ==========================================
# 6. TRAIN AND EVALUATE MODELS
# ==========================================

print("\nTraining models...\n")

results = {}

# Convert test target from log scale to USD
y_test_usd = np.expm1(y_test)

for name, model in models.items():

    print(f"Training {name}...")

    # Train model
    model.fit(X_train, y_train)

    # Predict in log scale
    predictions_log = model.predict(X_test)

    # Convert predictions back to USD
    predictions = np.expm1(predictions_log)

    # Evaluation metrics in USD
    mae = mean_absolute_error(
        y_test_usd,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_usd,
            predictions
        )
    )

    r2 = r2_score(
        y_test_usd,
        predictions
    )

    # Evaluation metric in log scale
    log_mae = mean_absolute_error(
        y_test,
        predictions_log
    )

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Log MAE": log_mae
    }

    print(f"\n{name}")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R2: {r2:.4f}")
    print(f"Log-scale MAE: {log_mae:.4f}")

    print("-" * 40)


# ==========================================
# 7. COMPARE MODELS
# ==========================================

print("\nMODEL COMPARISON")

results_df = pd.DataFrame(results).T

print(results_df)


# ==========================================
# 8. SELECT BEST MODEL
# ==========================================

# Select model with lowest log-scale MAE
best_model_name = results_df["Log MAE"].idxmin()

best_model = models[best_model_name]

print("\nBest Model:", best_model_name)


# ==========================================
# 9. SAVE BEST MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest model saved to models/best_model.pkl")

print("\nTraining completed successfully!")
