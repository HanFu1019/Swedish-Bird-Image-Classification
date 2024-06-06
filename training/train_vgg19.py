import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Import tqdm for the progress bar
import os
from torch.utils.data import random_split
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator
import numpy as np
import torchvision.models as models
import json



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

data_dir = '../Yolo_Crop/SWEDEN_crops'  # Replace this with the path to your data

save_dir = './saved_models/vgg19'
os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)



#indices = list(range(len(dataset)))
#train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=[label for _, label in dataset], random_state=42)   

with open('C:/Users/43477/Desktop/Training/saved_models/train_ind.json', 'r') as f:
    train_indices = json.load(f)

with open('C:/Users/43477/Desktop/Training/saved_models/test_ind.json', 'r') as f:
    test_indices = json.load(f)

    
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# DataLoader
batch_size = 16

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

num_classes = len(dataset.classes)
print(num_classes)


model = models.vgg19(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
num_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

model.to(device)

start_epoch = -1
RESUME = False

if RESUME:
    path_checkpoint = "./saved_models/vgg19/model_epoch1.pt" 
    checkpoint = torch.load(path_checkpoint)

    model.load_state_dict(checkpoint['net']) 

    optimizer.load_state_dict(checkpoint['optimizer']) 
    start_epoch = checkpoint['epoch']
    

num_epochs = 20

for epoch in range(start_epoch + 1, start_epoch + 1+num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{start_epoch + 1+num_epochs}')

    for step, (images, labels) in enumerate(progress_bar, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)
        progress_bar.set_postfix(loss=running_loss/len(progress_bar), accuracy=100 * correct_predictions / total_predictions)

    accuracy = 100 * correct_predictions / total_predictions
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    
    checkpoint = {
        "net": model.state_dict(),
        'optimizer':optimizer.state_dict(),
        "epoch": epoch
    }
    
    
    model_path = os.path.join(save_dir, f'model_epoch{epoch+1}.pt')
    #torch.save(model.state_dict(), model_path)
    torch.save(checkpoint, model_path)
    
    
    
    
