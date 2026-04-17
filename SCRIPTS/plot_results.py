import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from load_data import get_train_val_test_splits
from dataset import FoodDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "OUTPUT")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model.pth")


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
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model.to(DEVICE)


def predict_test_set(batch_size=32):
    _, _, test_dataset = get_train_val_test_splits(test_size=0.2, val_size=0.1, seed=42)
    dataloader = DataLoader(
        FoodDataset(test_dataset, get_transforms()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = load_model()
    model.eval()

    labels = []
    predictions = []

    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(DEVICE)
            preds = model(images).squeeze().cpu().numpy()
            labels.extend(target.numpy())
            predictions.extend(preds)

    return np.array(labels, dtype=float), np.array(predictions, dtype=float)


def compute_metrics(labels, predictions):
    errors = predictions - labels
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    median_error = np.median(errors)
    iqr = np.percentile(errors, 75) - np.percentile(errors, 25)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Median error": median_error,
        "Error IQR": iqr,
    }


def plot_error_distribution(labels, predictions, metrics):
    errors = predictions - labels
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color="#4c72b0", alpha=0.8)
    plt.axvline(0, color="red", linestyle="--", label="No error")
    plt.xlabel("Prediction error (predicted - actual)")
    plt.ylabel("Count")
    plt.title("Error Distribution")
    plt.grid(alpha=0.3)
    plt.legend()

    text = (
        f"MAE: {metrics['MAE']:.1f}\n"
        f"RMSE: {metrics['RMSE']:.1f}\n"
        f"Median error: {metrics['Median error']:.1f}\n"
        f"IQR: {metrics['Error IQR']:.1f}"
    )
    plt.gcf().text(0.02, 0.95, text, fontsize=10, va="top")

    save_path = os.path.join(OUTPUT_DIR, "error_distribution_explained.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels, predictions = predict_test_set()
    metrics = compute_metrics(labels, predictions)

    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

    path = plot_error_distribution(labels, predictions, metrics)

    print("Saved plot:")
    print(f"- {path}")


if __name__ == "__main__":
    main()
