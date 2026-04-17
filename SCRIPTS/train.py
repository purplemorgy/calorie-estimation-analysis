import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

from load_data import get_train_val_test_splits
from dataset import FoodDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "OUTPUT")


def get_model():
    # modern pretrained weights API (fixes warning)
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # replace final layer for regression
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model.to(DEVICE)


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(images).squeeze()
            loss = criterion(preds, labels)

            total_loss += loss.item()

    return total_loss / len(loader)


def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------
    # Load dataset and create splits
    # -----------------------
    train_dataset, val_dataset, _ = get_train_val_test_splits(test_size=0.2, val_size=0.1, seed=42)

    train_loader = DataLoader(
        FoodDataset(train_dataset, get_transforms()),
        batch_size=32,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        FoodDataset(val_dataset, get_transforms()),
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # -----------------------
    # Model / Loss / Optim
    # -----------------------
    model = get_model()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # -----------------------
    # Early stopping
    # -----------------------
    best_val_loss = float("inf")
    patience = 3
    counter = 0
    num_epochs = 10

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(images).squeeze()
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, criterion)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")

        # -----------------------
        # Save best model + early stopping
        # -----------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0

            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pth"))
            print("Saved best model.")
        else:
            counter += 1
            print(f"No improvement. Patience: {counter}/{patience}")

            if counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    train()