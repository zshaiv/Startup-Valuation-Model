
import pandas as pd
from pathlib import Path
import numpy as np

# Dataset location
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

# Load funding rounds dataset
print("Loading funding rounds dataset...")

funding = pd.read_csv(
    DATA_DIR / "funding_rounds.csv",
    low_memory=False
)

print("Original dataset shape:", funding.shape)

# Check available columns
print("\nColumns in dataset:")
print(funding.columns.tolist())

# Filter rows with valid post-money valuation
funding = funding[
    pd.to_numeric(
        funding["post_money_valuation_usd"],
        errors="coerce"
    ) > 0
].copy()

print("\nRows with valid post-money valuation:", len(funding))

# Convert valuation and funding amount to numeric
funding["post_money_valuation_usd"] = pd.to_numeric(
    funding["post_money_valuation_usd"],
    errors="coerce"
)

funding["raised_amount_usd"] = pd.to_numeric(
    funding["raised_amount_usd"],
    errors="coerce"
)

# Display summary statistics
print("\nPost-money valuation statistics:")
print(funding["post_money_valuation_usd"].describe())

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save cleaned dataset
output_file = OUTPUT_DIR / "valuation_data.csv"

funding.to_csv(output_file, index=False)

print(f"\nCleaned dataset saved to: {output_file}")
print("Data preparation completed!")


# ==========================================
# PHASE 2: Inspect company-level information
# ==========================================

print("\nLoading company dataset...")

companies = pd.read_csv(
    DATA_DIR / "objects.csv",
    low_memory=False
)

print("Company dataset shape:", companies.shape)

print("\nCompany dataset columns:")
print(companies.columns.tolist())

print("\nSample company records:")
print(companies.head(5).to_string())

print("\nMissing values in important company columns:")

for column in ["id", "name", "category_code", "founded_at", "status"]:
    if column in companies.columns:
        print(
            column,
            "missing:",
            companies[column].isnull().sum()
        )


# ==========================================
# PHASE 3: Merge funding rounds with company data
# ==========================================

print("\nMerging funding and company datasets...")

# Keep only company columns useful for initial modeling
company_features = companies[
    [
        "id",
        "name",
        "category_code",
        "founded_at",
        "status",
        "relationships",
        "funding_total_usd",
        "country_code",
    ]
].copy()

# Merge using company ID
merged = funding.merge(
    company_features,
    left_on="object_id",
    right_on="id",
    how="left"
)

print("Merged dataset shape:", merged.shape)

print("\nMissing values in merged dataset:")
print(
    merged[
        [
            "category_code",
            "founded_at",
            "relationships",
            "funding_total_usd",
            "country_code",
        ]
    ].isnull().sum()
)

# Save merged dataset
merged.to_csv(
    OUTPUT_DIR / "merged_valuation_data.csv",
    index=False
)

print("\nMerged dataset saved successfully!")


# ==========================================
# PHASE 4: Feature engineering
# ==========================================

print("\nCreating modeling features...")

# Convert dates
merged["funded_at"] = pd.to_datetime(
    merged["funded_at"],
    errors="coerce"
)

merged["founded_at"] = pd.to_datetime(
    merged["founded_at"],
    errors="coerce"
)

# Extract years
merged["funding_year"] = merged["funded_at"].dt.year
merged["founding_year"] = merged["founded_at"].dt.year

# Calculate company age at funding
merged["company_age"] = (
    merged["funding_year"] - merged["founding_year"]
)

# Clean invalid company ages
merged.loc[
    merged["company_age"] < 0,
    "company_age"
] = None

# Target: log-transformed post-money valuation
merged["log_valuation"] = np.log1p(
    merged["post_money_valuation_usd"]
)

# Select modeling features
feature_columns = [
    "funding_round_type",
    "funding_round_code",
    "raised_amount_usd",
    "category_code",
    "company_age",
    "relationships",
    "country_code",
]

model_data = merged[
    ["object_id"] + feature_columns +
    ["post_money_valuation_usd", "log_valuation"]
].copy()

# Remove rows with missing target
model_data = model_data.dropna(
    subset=["post_money_valuation_usd", "log_valuation"]
)

print("\nModeling dataset shape:", model_data.shape)

print("\nFeature columns:")
print(feature_columns)

print("\nSample modeling data:")
print(model_data.head())

# Save modeling dataset
model_data.to_csv(
    OUTPUT_DIR / "modeling_data.csv",
    index=False
)

print("\nModeling dataset saved!")
