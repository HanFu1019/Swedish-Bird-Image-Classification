import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm  # Import tqdm for the progress bar
import os
import torchvision.models as models
import json

#GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

data_dir = '../model/SWEDEN_crops/SWEDEN_crops'  # Replace this with the path to your data

save_dir = '../model/saved_models/ensemble'
os.makedirs(save_dir, exist_ok=True)

# Transforms for the input data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])

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
# Define Ensemble model
class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, input):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        for param in modelA.parameters():
            param.requires_grad = False
        self.modelB = modelB
        for param in modelB.parameters():
            param.requires_grad = False
 
        self.modelA.fc = nn.Identity() 
        self.modelB.classifier = nn.Identity() 

        self.fc1 = nn.Linear(input, 507)

    def forward(self, x):
        out1 = self.modelA(x)
        out1 = out1.view(out1.size(0),-1)
        out2 = self.modelB(x)
        out2 = out2.view(out2.size(0),-1)

        out = torch.cat((out1,out2), dim=1)

        out = self.fc1(out)
        return torch.softmax(out, dim=1)

# Load model A
model1 = models.resnet101()
num_features1 = model1.fc.in_features
model1.fc = nn.Linear(num_features1, num_classes)

path_checkpoint1 = "./saved_models/resnet101/resnet101.pt" 
model1.load_state_dict(torch.load(path_checkpoint1))

# Load model B
model2 = models.densenet169()
num_features2 = model2.classifier.in_features
model2.classifier = nn.Linear(num_features2, num_classes)

path_checkpoint2 = "./saved_models/densenet169/densenet169.pt" 
checkpoint = torch.load(path_checkpoint2)
model2.load_state_dict(checkpoint['net'])

# Create ensemble model
model = MyEnsemble(model1, model2, num_features1+num_features2)

# Load model checkpoint if you want to continue training
#model_path = '../model/saved_models/ensemble/ensemble.pt'  # Specify the path to the saved model
#model.load_state_dict(torch.load(model_path))

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0002)

# Training Loop
num_epochs = 30

for epoch in range(0, num_epochs):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
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
        
    # Print epoch loss
    accuracy = 100 * correct_predictions / total_predictions
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
    
    # Save epoch checkpoint
    model_path = os.path.join(save_dir, f'model_epoch{epoch+1}.pt')
    torch.save(model.state_dict(), model_path)