
import pandas as pd
from pathlib import Path

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
