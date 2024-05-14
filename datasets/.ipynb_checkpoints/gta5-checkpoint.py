from torch.utils.data import Dataset
import os
import posixpath
import numpy as np
import torch
# TODO: implement here your custom dataset class for Cityscapes
from torchvision.io import read_image 
from torchvision import transforms
from PIL import Image
def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image
def convert_tensor_to_image(tensor):
    image = tensor.transpose((1, 2, 0))
    return image
class GTA5(Dataset):
    def __init__(self,annotations_dir,images_dir,transform=None, target_transform=None):
        super(GTA5, self).__init__()
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.target_transform = target_transform
        self.map_index_to_image = []
        self.map_index_to_annotation = []
        self.generate_map_images(path=self.images_dir)
        self.generate_map_annotations(path=self.annotations_dir)
    def generate_map_images(self,path='.'):
        for entry in os.listdir(path):
            full_path = posixpath.join(path, entry)
            if os.path.isdir(full_path):
                self.generate_map_images(full_path)
            else:
                #self.map_index_to_image[counter]=full_path
                self.map_index_to_image.append(full_path)
    def generate_map_annotations(self,path='.'):
        for entry in os.listdir(path):
            full_path = posixpath.join(path, entry)
            if os.path.isdir(full_path):
                self.generate_map_annotations(full_path)
            else:
                self.map_index_to_annotation.append(full_path)

    def __getitem__(self, idx):
        image=""
        annotation=""
        image_path = self.map_index_to_image[idx]
        image = read_image(image_path)
        #return last two elements of the path
        path=image_path.split('/')
        image_name = posixpath.join(path[-2],path[-1])
        
        #annotation_path = posixpath.join(self.annotations_dir, image_name.replace("_leftImg8bit.png","_gtFine_labelTrainIds.png"))
        annotation_path = self.map_index_to_annotation[idx]
        annotation = read_image(annotation_path)#[0:3,:,:]
        if self.transform:
             image = self.transform(image)
             annotation=self.transform(annotation)
             annotation= torch.tensor(self.transform(annotation),dtype=torch.uint8)
        return image, annotation

    def __len__(self):
        return len(self.map_index_to_image)
 
