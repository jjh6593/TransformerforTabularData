import torch
from torch.utils.data import Dataset, DataLoader, random_split


class TabularDataset(Dataset):
    def __init__(self, features, answer, mask_ratio=0.1, mask_value=0):
        self.features = features
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.answer = answer

        self.masked_features = []
        self.masks = []
        for feature in self.features:
            mask = (torch.rand(feature.size()) < self.mask_ratio).float()
            masked_feature = feature * (1 - mask) + self.mask_value * mask
            self.masked_features.append(masked_feature)
            self.masks.append(mask)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.masked_features[idx], self.masks[idx], self.features[idx], self.answer[idx]


def create_dataloaders(dataset, batch_size, train_split):
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
