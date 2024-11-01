import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm  # Import tqdm for the progress bar
import os
import torchvision.models as models
import json

# GPU or CPu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

data_dir = '../model/SWEDEN_crops/SWEDEN_crops'  # Replace this with the path to your data

# Transforms for the input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

# Dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

num_classes = len(dataset.classes)

# Load train-test indice splits
with open('../model/train_ind.json', 'r') as f:
    train_indices = json.load(f)

with open('../model/test_ind.json', 'r') as f:
    test_indices = json.load(f)

train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

# DataLoader
batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

# Initialize model, loss, and optimizer
num_classes = len(dataset.classes)
model = models.vgg19(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
# Define classification head
num_ftrs = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

# Load model checkpoint
path_checkpoint = "../model/sweden_models/vgg19/vgg19.pt" 
checkpoint = torch.load(path_checkpoint)
model.load_state_dict(checkpoint['net']) 

model.to(device)
model.eval()  # Set the model to evaluation mode

criterion = nn.CrossEntropyLoss()

# Define top 5 prediction accuracy
def top_5_accuracy(outputs, targets):
    # Get the indices of the top 5 predictions for each sample
    _, predicted = torch.topk(outputs, 5, dim=1)
    
    # Initialize a variable to store the count of correct predictions
    correct = 0
    
    # Iterate over each sample in the batch
    for i in range(targets.size(0)):
        # Check if the true label is among the top 5 predicted classes
        if targets[i] in predicted[i]:
            # If yes, increment the count of correct predictions
            correct += 1
    
    # Compute the top-5 accuracy for the batch
    accuracy = correct 
    
    return accuracy

# Choose between top_5 and top_1 accuracy
top_5 = False

# Test the model on the test dataset
correct_predictions = 0
total_predictions = 0
running_loss = 0.0
with torch.no_grad():
    progress_bar = tqdm(test_loader)
    
    for step, (images, labels) in enumerate(progress_bar, 1):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()

        # Choose between top_5 and top_1 accuracy
        if top_5 == True:
            correct_predictions += top_5_accuracy(outputs, labels) 
        else:
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

        progress_bar.set_postfix(loss=running_loss/len(progress_bar), accuracy=100 * correct_predictions / total_predictions)

# Print test accuracy
test_accuracy = 100 * correct_predictions / total_predictions
print(f'Test Accuracy: {test_accuracy}%')