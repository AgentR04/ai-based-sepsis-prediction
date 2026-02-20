import pandas as pd

df = pd.read_csv("data/raw/vitals_hourly.csv")
print(df["vital"].value_counts())
df.groupby("vital")["value"].agg(["min", "max"])
