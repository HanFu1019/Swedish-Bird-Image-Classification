from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import os
from sklearn.model_selection import StratifiedKFold
import json

data_dir = '../model/SWEDEN_crops/SWEDEN_crops'  # Replace this with the path to your data

# Transforms for the input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Get the indices of the dataset, a.k.a. the number of classes
indices = list(range(len(dataset)))

# Split them using stratified spliting, so the splits are balanced
train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=[label for _, label in dataset], random_state=42)

# Save the indices splits
with open('train_ind.json', 'w') as f:
    json.dump(train_indices, f)

with open('test_ind.json', 'w') as f:
    json.dump(test_indices, f)
