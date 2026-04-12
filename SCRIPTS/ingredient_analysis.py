from load_data import load_food_dataset
import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import Counter

OUTPUT_DIR = "../OUTPUT"


def extract_ingredient_counts(df):
    ingredient_counter = Counter()
    num_ingredients_per_plate = []

    for ingredients_list in df["ingredients"]:
        if ingredients_list is None:
            continue

        num_ingredients_per_plate.append(len(ingredients_list))

        for ingredient in ingredients_list:
            name = ingredient.get("name", "unknown")
            ingredient_counter[name] += 1

    return ingredient_counter, num_ingredients_per_plate


def plot_top_ingredients(counter, top_n=20):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    most_common = counter.most_common(top_n)
    names = [x[0] for x in most_common]
    counts = [x[1] for x in most_common]

    plt.figure()
    plt.barh(names, counts)
    plt.title("Top Ingredients")
    plt.xlabel("Count")
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/top_ingredients.png")
    plt.close()
    print(f"Saved top_ingredients.png")


def plot_num_ingredients_distribution(num_ingredients):
    plt.figure()
    plt.hist(num_ingredients, bins=20)
    plt.title("Number of Ingredients per Plate")
    plt.xlabel("Count")
    plt.ylabel("Frequency")

    plt.savefig(f"{OUTPUT_DIR}/ingredient_count_distribution.png")
    plt.close()
    print(f"Saved ingredient_count_distribution.png")


if __name__ == "__main__":
    df = load_food_dataset()

    counter, num_ingredients = extract_ingredient_counts(df)

    plot_top_ingredients(counter)
    plot_num_ingredients_distribution(num_ingredients)