import matplotlib.pyplot as plt
from load_data import load_food_dataset
import os

OUTPUT_DIR = "../OUTPUT"


def plot_macro_distributions(df):
    macros = [
        "total_calories",
        "total_fat",
        "total_carb",
        "total_protein"
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for macro in macros:
        plt.figure()
        plt.hist(df[macro], bins=50)
        plt.title(f"Distribution of {macro}")
        plt.xlabel(macro)
        plt.ylabel("Frequency")

        save_path = f"{OUTPUT_DIR}/{macro}_distribution.png"
        plt.savefig(save_path)
        plt.close()

        print(f"Saved {macro}_distribution.png")


if __name__ == "__main__":
    df = load_food_dataset()
    plot_macro_distributions(df)