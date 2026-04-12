from datasets import load_dataset
import pandas as pd

def load_food_dataset():
    dataset = load_dataset("mmathys/food-nutrients")

    df = dataset["test"].to_pandas()
    
    return df


if __name__ == "__main__":
    df = load_food_dataset()
    print(df.head())
    print(df.columns)