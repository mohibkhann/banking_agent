
import pandas as pd

df = pd.read_csv('C:/Users/mohib.alikhan/Desktop/Banking-Agent/Banking_Data.csv')
df["date"] = pd.to_datetime(df["date"])

# Check if client 430 has data in September 2023
mask = (df["client_id"] == 430) & (df["date"].between("2023-09-01", "2023-09-30"))
print(df[mask])