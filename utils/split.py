import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_val_split(csv_path, val_size=0.1):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=val_size, stratify=df["label"], random_state=42)
    train_csv = "data/train_split.csv"
    val_csv = "data/val_split.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print(f"Created {train_csv} and {val_csv}")
    return train_csv, val_csv
