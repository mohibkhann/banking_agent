import pandas as pd

# 1. Load your raw transactional data
client_data = pd.read_csv(
    "/Users/mohibalikhan/Desktop/banking-agent/banking_agent/Banking_Data.csv",
    parse_dates=["date", "acct_open_date"]
)

# 2. Define which columns are PII and should be removed
pii_cols = [
    "transaction_id",  # internal key
    "client_id",       # direct client identifier
    "card_id",         # individual card number
    "acct_open_date",  # exact account inception
    "merchant_city",   # location detail
    "merchant_state",
    "zip",             # postal code
]

# 3. Drop PII
overall_data = client_data.drop(columns=pii_cols)

# 4. Derive any time-based or categorical flags you need
overall_data = overall_data.assign(
    day_name    = client_data["date"].dt.day_name(),
    month       = client_data["date"].dt.to_period("M").astype(str),
    txn_hour    = client_data["txn_time"].str.slice(0,2).astype(int),
)
overall_data["is_night_txn"] = overall_data["txn_hour"].between(22, 23) | overall_data["txn_hour"].between(0, 6)
overall_data["is_weekend"]   = client_data["date"].dt.dayofweek >= 5

# 5. (Optional) Reorder columns or cast types
#   —for example, ensure boolean flags are bool dtype:
for col in ["is_night_txn", "is_weekend", "use_chip", "card_on_dark_web"]:
    overall_data[col] = overall_data[col].astype(bool)

# 6. Inspect
print(overall_data.shape)
print(overall_data.dtypes)
print(overall_data.head())


overall_data.to_csv("overall_data.csv", index = False)