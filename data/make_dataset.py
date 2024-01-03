import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import pickle


train_images_list = []
train_targets_list = []
for num in range(0, 6):
    train_images_0 = torch.load(f"data/raw/corruptmnist/train_images_{num}.pt")
    train_images_0 = train_images_0.unsqueeze(1)  # Add channel dimension to fit CNN
    train_targets_0 = torch.load(f"data/raw/corruptmnist/train_target_{num}.pt")
    train_images_list.append(train_images_0)
    train_targets_list.append(train_targets_0)


# Load training data
train_images = torch.cat(train_images_list, dim=0)
train_targets = torch.cat(train_targets_list, dim=0)

mean = train_images.mean(axis=(0, 2, 3))
std = train_images.std(axis=(0, 2, 3))
normalize = transforms.Normalize(mean=mean, std=std)
train_images = normalize(train_images)


# Load testing data
test_images = torch.load("data/raw/corruptmnist/test_images.pt")
test_images = test_images.unsqueeze(1)
test_targets = torch.load("data/raw/corruptmnist/test_target.pt")

mean = train_images.mean(axis=(0, 2, 3))
std = train_images.std(axis=(0, 2, 3))
normalize = transforms.Normalize(mean=mean, std=std)
test_images = normalize(test_images)

# Create TensorDatasets
train_dataset = TensorDataset(train_images, train_targets)
test_dataset = TensorDataset(test_images, test_targets)

# store train_dataset into file
with open("data/processed/train_dataset.pkl", "wb") as file:
    pickle.dump(train_dataset, file)

# store test_dataset into file
with open("data/processed/test_dataset.pkl", "wb") as file:
    pickle.dump(test_dataset, file)
