
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/best_model.pkl")


def predict_valuation(
    raised_amount_usd,
    company_age,
    relationships,
    funding_round_type,
    funding_round_code,
    category_code,
    country_code
):
    input_data = pd.DataFrame([{
        "raised_amount_usd": raised_amount_usd,
        "company_age": company_age,
        "relationships": relationships,
        "funding_round_type": funding_round_type,
        "funding_round_code": funding_round_code,
        "category_code": category_code,
        "country_code": country_code
    }])

    # Predict log valuation
    predicted_log_valuation = model.predict(input_data)[0]

    # Convert back to USD
    predicted_valuation = np.expm1(
        predicted_log_valuation
    )

    return predicted_valuation


if __name__ == "__main__":

    valuation = predict_valuation(
        raised_amount_usd=5000000,
        company_age=3,
        relationships=10,
        funding_round_type="venture",
        funding_round_code="A",
        category_code="software",
        country_code="USA"
    )

    print("\nPredicted Startup Valuation:")
    print(f"${valuation:,.2f}")
