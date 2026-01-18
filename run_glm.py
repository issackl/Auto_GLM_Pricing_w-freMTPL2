import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

CSV_PATH = "model_policy.csv"

df = pd.read_csv(CSV_PATH)

# Required columns
required = ["Exposure", "ClaimNb"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")

# Make sure numeric
df["Exposure"] = pd.to_numeric(df["Exposure"], errors="coerce")
df["ClaimNb"] = pd.to_numeric(df["ClaimNb"], errors="coerce").fillna(0)

# Create LogExp if not present
if "LogExp" not in df.columns:
    df["LogExp"] = np.where(df["Exposure"] > 0, np.log(df["Exposure"]), np.nan)
else:
    df["LogExp"] = pd.to_numeric(df["LogExp"], errors="coerce")

# Basic filter
df = df[(df["Exposure"] > 0) & np.isfinite(df["LogExp"])].copy()

# Choose predictors (uses bands if you created them, otherwise uses raw columns)
candidate_predictors = [
    "DriverAgeBand", "VehAgeBand", "BonusMalusBand",  # banded categoricals
    "VehBrand", "VehGas", "Area",                     # common categoricals
    "VehPower", "VehAge", "DrivAge", "BonusMalus"     # numeric fallbacks
]
predictors = [c for c in candidate_predictors if c in df.columns]
if not predictors:
    raise ValueError("No predictor columns found in the CSV. Add some rating variables to your export.")

# Build formula with C() around categorical/text columns
terms = []
for c in predictors:
    if df[c].dtype == "object":
        terms.append(f"C({c})")
    else:
        terms.append(c)

freq_formula = "ClaimNb ~ " + " + ".join(terms)
print("Frequency formula:", freq_formula)

# Frequency GLM: Poisson with offset log(exposure)
freq_model = smf.glm(
    formula=freq_formula,
    data=df,
    family=sm.families.Poisson(),
    offset=df["LogExp"]
).fit()
print("\nFREQUENCY MODEL")
print(freq_model.summary())

# Predicted claims and frequency
df["pred_claims"] = freq_model.predict(df, offset=df["LogExp"])
df["pred_freq"] = df["pred_claims"] / df["Exposure"]

# Severity GLM setup (needs AvgSev)
if "AvgSev" not in df.columns:
    raise ValueError("AvgSev column not found. Export the policy table that includes AvgSev.")

df["AvgSev"] = pd.to_numeric(df["AvgSev"], errors="coerce")

sev_df = df[(df["ClaimNb"] > 0) & (df["AvgSev"] > 0)].copy()
if sev_df.empty:
    raise ValueError("No claimant rows found for severity model. Check ClaimNb and AvgSev.")

# Cap extreme severities to stabilize fit
cap = sev_df["AvgSev"].quantile(0.995)
sev_df["AvgSev_capped"] = np.minimum(sev_df["AvgSev"], cap)

sev_terms = []
for c in predictors:
    if c not in sev_df.columns:
        continue
    if sev_df[c].dtype == "object":
        sev_terms.append(f"C({c})")
    else:
        sev_terms.append(c)

sev_formula = "AvgSev_capped ~ " + " + ".join(sev_terms)
print("\nSeverity formula:", sev_formula)

sev_model = smf.glm(
    formula=sev_formula,
    data=sev_df,
    family=sm.families.Gamma(sm.families.links.log()),
    freq_weights=sev_df["ClaimNb"]  # weight by number of claims
).fit()
print("\nSEVERITY MODEL")
print(sev_model.summary())

# Predict severity and pure premium
df["pred_sev"] = sev_model.predict(df)
df["pred_pure_premium"] = df["pred_freq"] * df["pred_sev"]

# Output for Excel merge
id_col = "IDpol" if "IDpol" in df.columns else None
out_cols = ([id_col] if id_col else []) + ["pred_freq", "pred_sev", "pred_pure_premium"]
df[out_cols].to_csv("glm_predictions.csv", index=False)

print("\nSaved glm_predictions.csv")
print(df[out_cols].head(10))