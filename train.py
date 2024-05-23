# TODO: Define here your training and validation loops.
"""
from datasets.cityscapes import CityScapes
from torch.utils.data import DataLoader
from models.deeplabv2.deeplabv2 import * 
import matplotlib.pyplot as plt
from torchvision import transforms
dataset_path='C:/Users/Federico/Desktop/Universita/1_Magistrale/2_semestre/MLDL/proj/MLDL_project/datasets/Cityscapes/Cityscapes/Cityspaces/'
annotation_train=dataset_path+'gtFine/train'
image_train=dataset_path+'images/train'

annotation_val=dataset_path+'gtFine/val'
image_val=dataset_path+'images/val'

cityscapes_train = CityScapes(annotations_dir=annotation_train, images_dir=image_train,transform=transforms.Resize(size = (512,1024)))
cityscapes_val = CityScapes(annotations_dir=annotation_val, images_dir=image_val,transform=transforms.Resize(size = (512,1024)))
print(cityscapes_train.__len__())
print(cityscapes_val.__len__())
index=55
print(cityscapes_train.map_index_to_image[index])
print(cityscapes_train.map_index_to_annotation[index])
#cityscapes_train.list_files_recursive(path=image_train)
image,annotation=cityscapes_train.__getitem__(index)
fig, axes = plt.subplots(2, 1, figsize=(1, 10))
axes[0].imshow(image)
axes[1].imshow(annotation,cmap='gray')
plt.show()
from PIL import Image
import numpy as np



import torch.optim as optim

# Set up data loaders
train_loader = DataLoader(cityscapes_train, batch_size=4, shuffle=True, num_workers=4)
val_loader = DataLoader(cityscapes_val, batch_size=4, shuffle=False, num_workers=4)

# Initialize the model
model = get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path='C:/Users/Federico/Desktop/Universita/1_Magistrale/2_semestre/MLDL/proj/MLDL_project/models/deeplab_resnet_pretrained_imagenet.pth')
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Adjust if your model outputs different dimensions
#optimizer = optim.Adam(model.optim_parameters(lr=0.0001))

# Modify here to ensure all parameters are captured correctly
def optim_parameters(model, lr):
    return [
        {'params': [p for p in model.get_1x_lr_params_no_scale()], 'lr': lr},
        {'params': [p for p in model.layer6.parameters()], 'lr': 10 * lr}  # Directly accessing parameters of layer6
    ]

optimizer = optim.Adam(optim_parameters(model, 0.0001))

# Training loop
def train(num_epochs, model, loaders, criterion, optimizer, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        print(epoch)
        for data in loaders['train']:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            print("ciao")
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loaders['train'])}")

# Validation loop
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
    print(f"Validation Loss: {total_loss / len(loader)}")

# Running the training and validation
loaders = {
    'train': train_loader,
    'validate': val_loader
}
train(1, model, loaders, criterion, optimizer)  # Train for 10 epochs as an example
validate(model, val_loader, criterion)
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from datasets.cityscapes import CityScapes
from models.deeplabv2.deeplabv2 import get_deeplab_v2
import numpy as np
from PIL import Image
import torch.nn as nn

def optim_parameters(model, lr):
    return [
        {'params': [p for p in model.get_1x_lr_params_no_scale()], 'lr': lr},
        {'params': [p for p in model.layer6.parameters()], 'lr': 10 * lr}
    ]

def train(num_epochs, model, loaders, criterion, optimizer, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        print(epoch)
        for batch_idx, (inputs, targets) in enumerate(loaders['train']):
            #print(data)
            inputs, targets = inputs.to(device), targets.to(device)
            print("Image shape:", inputs.shape)  # Ensure the shape is correct
            print("Label shape:", targets.shape)  # Check the shape of labels if needed
            inputs = inputs.float()
            targets = targets.squeeze()
            print("c8ao")
            outputs = model(inputs)
            optimizer.zero_grad()
            print("ciao")
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loaders['train'])}")

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss / len(loader)}")

def main():
    dataset_path = 'C:/Users/Federico/Desktop/Universita/1_Magistrale/2_semestre/MLDL/proj/MLDL_project/datasets/Cityscapes/Cityscapes/Cityspaces/'
    annotation_train = dataset_path + 'gtFine/train'
    image_train = dataset_path + 'images/train'
    annotation_val = dataset_path + 'gtFine/val'
    image_val = dataset_path + 'images/val'
    resize_transform = transforms.Resize(interpolation=transforms.InterpolationMode.NEAREST_EXACT,size = (512,1024))
    # Setup device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cityscapes_train = CityScapes(annotations_dir=annotation_train, images_dir=image_train,transform=resize_transform)
    cityscapes_val = CityScapes(annotations_dir=annotation_val, images_dir=image_val,transform=resize_transform)
    """
    print(cityscapes_train.__len__())
    print(cityscapes_val.__len__())
    index=55
    print(cityscapes_train.map_index_to_image[index])
    print(cityscapes_train.map_index_to_annotation[index])
    #cityscapes_train.list_files_recursive(path=image_train)
    image,annotation=cityscapes_train.__getitem__(index)
    """
    fig, axes = plt.subplots(2, 1, figsize=(1, 10))
    axes[0].imshow(image)
    axes[1].imshow(annotation,cmap='gray')
    plt.show()
"""
    

    train_loader = DataLoader(cityscapes_train, batch_size=16, shuffle=True)
    val_loader = DataLoader(cityscapes_val, batch_size=16, shuffle=True)

    model = get_deeplab_v2(num_classes=19, pretrain=True, pretrain_model_path='C:/Users/Federico/Desktop/Universita/1_Magistrale/2_semestre/MLDL/proj/MLDL_project/models/deeplab_resnet_pretrained_imagenet.pth')
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(optim_parameters(model, 0.0001))

    loaders = {
        'train': train_loader,
        'validate': val_loader
    }

    #train(1, model, loaders, criterion, optimizer, torch.device("cuda" if torch.cuda.is_available() else "cpu"))  # Train for 1 epoch as an example
    #validate(model, val_loader, criterion, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
"""


if __name__ == '__main__':
    main()
