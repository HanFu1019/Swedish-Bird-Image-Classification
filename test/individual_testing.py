import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm  # Import tqdm for the progress bar
import os
import numpy as np
import torchvision.models as models
import json
import matplotlib.pyplot as plt

# GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


data_dir = '../model/SWEDEN_crops/SWEDEN_crops'  # Replace this with the path to your data

# Transforms for the input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)

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
model = models.densenet169(weights='IMAGENET1K_V1')
for param in model.parameters():
    param.requires_grad = False
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, num_classes)

# Load model checkpoint
path_checkpoint = "../model/sweden_models/densenet169/densenet169.pt" 
checkpoint = torch.load(path_checkpoint)
model.load_state_dict(checkpoint['net']) 

model.to(device)
model.eval()  # Set the model to evaluation mode

criterion = nn.CrossEntropyLoss()

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


def imshow(img, title):
    # Unnormalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img.numpy().transpose((1, 2, 0))  # Convert from Tensor to numpy array and rearrange dimensions
    img = std * img + mean  # Unnormalize
    img = np.clip(img, 0, 1)  # Clip values to be between 0 and 1
    
    plt.imshow(img)
    plt.title(title)
    plt.show()

# Plot 4 predictions
c=0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Get the predicted labels
        _, predicted = torch.topk(outputs, 5, dim=1)
        
        # Move images to CPU for plotting
        images = images.cpu()
        for i in range(len(images)):
            if labels[i] not in predicted[i]:
                img = images[i]
                true_label = dataset.classes[labels[i]]
                #predicted_label = dataset.classes[predicted[i]]
                title = f'True: {true_label}'#, Predicted: {predicted_label}'
                imshow(img, title)
                c = c+1
                if c >= 4:
                    break
        # Plot the images along with the true and predicted labels
        