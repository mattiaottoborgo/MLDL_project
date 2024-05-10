# TODO: Define here your training and validation loops.
#models for normal jupyter 
#from datasets.cityscapes import CityScapes
#from models.bisenet.build_bisenet import BiSeNet
#from utils_semantic_segmentation.utils import poly_lr_scheduler

from datasets.cityscapes import CityScapes
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from models.bisenet.build_bisenet import BiSeNet
from utils import poly_lr_scheduler

def mIoU(pred_mask, mask, smooth=1e-10, n_classes=19):
    with torch.no_grad():
        pred_mask = F.softmax(pred_mask, dim=1)
        pred_mask = torch.argmax(pred_mask, dim=1)
        pred_mask = pred_mask.contiguous().view(-1)
        mask = mask.contiguous().view(-1)

        iou_per_class = []
        for clas in range(0, n_classes): #loop per pixel class
            true_class = pred_mask == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: #no exist label in this loop
                iou_per_class.append(np.nan)
            else:
                #print("true class size",true_class.shape)
                #print("true label size",true_label.shape)
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                iou = (intersect + smooth) / (union +smooth)
                iou_per_class.append(iou)
        return np.nanmean(iou_per_class)
def pixel_accuracy(output, mask):
    with torch.no_grad():
        output = torch.argmax(F.softmax(output, dim=1), dim=1)
        correct = torch.eq(output, mask).int()
        accuracy = float(correct.sum()) / float(correct.numel())
    return accuracy

def convert_tensor_to_image(tensor):
    image = tensor.permute(1, 2, 0)
    return image
def train(model,optimizer, train_loader, criterion):
    model.train()
    running_loss = 0.0
    iou_score=0.0
    accuracy=0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.float()
        targets = targets.squeeze()
        #Compute prediction and loss
        outputs,_,_ = model(inputs)
        print(batch_idx)
        
        loss = loss_fn(outputs.to(dtype=torch.float32), targets.to(dtype=torch.int64))
        iou_score += mIoU(outputs.to(device), targets.to(device))
        accuracy += pixel_accuracy(outputs.to(device), targets.to(device))
        #BackPropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)

    train_loss = running_loss / len(train_loader)
    iou_score = iou_score / len(train_loader)
    accuracy = accuracy / len(train_loader)
    return train_loss,iou_score,accuracy

# Test loop
# calculate_label_prediction is a flag used to decide wether to calculate or not ground_truth and predicted tensor
def test(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    iou_score=0.0
    accuracy=0.0
    with torch.no_grad():
        for batch_idx,(inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            targets = targets.int()
            #Compute prediction and loss
            outputs = model(inputs)
            print(batch_idx)
            loss = loss_fn(outputs.to(dtype=torch.float32), targets.squeeze().to(dtype=torch.int64))
            iou_score += mIoU(outputs.to(device), targets.to(device))
            accuracy += pixel_accuracy(outputs.to(device), targets.to(device))
            test_loss += loss.item()
    test_loss = test_loss / len(test_loader)
    iou_score = iou_score / len(test_loader)
    accuracy = accuracy / len(test_loader)
    #test_accuracy = 100. * correct / total
    return test_loss,iou_score,accuracy


#dataset_path='/kaggle/input/cityscapes-polito/Cityscapes/Cityscapes/Cityspaces/'
dataset_path='datasets/Cityscapes/Cityscapes/Cityspaces/'
annotation_train=dataset_path+'gtFine/train'
image_train=dataset_path+'images/train'

annotation_val=dataset_path+'gtFine/val'
image_val=dataset_path+'images/val'
resize_transform = transforms.Resize(interpolation=transforms.InterpolationMode.NEAREST_EXACT,size = (512,1024))
# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
cityscapes_train = CityScapes(annotations_dir=annotation_train, images_dir=image_train,transform=resize_transform)
cityscapes_val = CityScapes(annotations_dir=annotation_val, images_dir=image_val,transform=resize_transform)

train_loader = DataLoader(cityscapes_train, batch_size=16, shuffle=True)
val_loader = DataLoader(cityscapes_val, batch_size=16, shuffle=True)

# Define the model and load it to the device
bisenet = BiSeNet(num_classes=19, context_path='resnet18')
bisenet.to(device)
optimizer = torch.optim.Adam(bisenet.parameters(), lr=0.001)
poly_lr_scheduler(optimizer, 0.01, 1, lr_decay_iter=1, max_iter=300, power=0.9)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
print(cityscapes_train.__len__())
epoch_beginning=0
epochs = 50

train_iou_list=[]
train_acc_list=[]
train_loss_list=[]

test_iou_list=[]
test_acc_list=[]
test_loss_list=[]

#uncomment this to load a model and continue training
""" 
version=37
path_weights=f"bisenet_epoch_{version}_weights.pth"
bisenet = BiSeNet(num_classes=19, context_path='resnet18')
bisenet.to(device)
#bisenet.load_state_dict(torch.load('/kaggle/input/cityscapes-polito/bisenet_epoch_9_weights.pth'))
bisenet.load_state_dict(torch.load(path_weights))
epoch_beginning=version+1
epochs = 50
"""


for epoch in range(epoch_beginning,epochs):
    train_loss,train_iou,train_acc=train(bisenet, optimizer, train_loader, loss_fn)
    train_iou_list.append(train_iou)
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    file_name='bisenet_epoch_'+str(epoch)+'_weights.pth'
    torch.save(bisenet.state_dict(),file_name)
    test_loss,test_iou,test_acc = test(bisenet, val_loader, loss_fn)
    test_iou_list.append(test_iou)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)
    print(f"Epoch n.{epoch} - Test loss: {test_loss}")