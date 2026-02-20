import pandas as pd

df = pd.read_csv('data/model/train_dataset.csv')
print(f'Dataset shape: {df.shape}')
print(f'\nLabel distribution:\n{df["label"].value_counts()}')
print(f'\nLabel percentages:\n{df["label"].value_counts(normalize=True)*100}')
print(f'\nColumn names (first 10):')
print(df.columns[:10].tolist())
