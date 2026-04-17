import torch
from torch.utils.data import Dataset

class FoodDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = item["image"]   # now PIL (correct after cast_column)
        image = self.transform(image)

        label = torch.tensor(item["total_calories"], dtype=torch.float32)

        return image, label