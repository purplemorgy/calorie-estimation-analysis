import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn as nn
import matplotlib.pyplot as plt

from load_data import get_train_val_test_splits
from dataset import FoodDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "OUTPUT")


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "model.pth"), map_location=DEVICE))
    return model.to(DEVICE)


def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _, _, test_dataset = get_train_val_test_splits(test_size=0.2, val_size=0.1, seed=42)
    loader = DataLoader(
        FoodDataset(test_dataset, get_transforms()),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model()
    model.eval()

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            preds = model(images).squeeze().cpu().numpy()
            preds_list.extend(preds)
            labels_list.extend(labels.numpy())

    mae = mean_absolute_error(labels_list, preds_list)
    rmse = np.sqrt(mean_squared_error(labels_list, preds_list))
    errors = np.array(preds_list) - np.array(labels_list)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"Median error: {np.median(errors):.2f}")
    print(f"Error IQR: {np.percentile(errors, 75) - np.percentile(errors, 25):.2f}")

    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"Median error: {np.median(errors):.2f}\n")
        f.write(f"Error IQR: {np.percentile(errors, 75) - np.percentile(errors, 25):.2f}\n")

    plt.figure()
    plt.scatter(labels_list, preds_list, alpha=0.5)
    plt.xlabel("Actual Calories")
    plt.ylabel("Predicted Calories")
    plt.title("Predicted vs Actual")
    plt.savefig(os.path.join(OUTPUT_DIR, "pred_vs_actual.png"))
    plt.close()

    plt.figure()
    plt.hist(errors, bins=50)
    plt.title("Error Distribution")
    plt.xlabel("Prediction Error")
    plt.savefig(os.path.join(OUTPUT_DIR, "error_distribution.png"))
    plt.close()


if __name__ == "__main__":
    evaluate()