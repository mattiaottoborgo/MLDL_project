from torch.utils.data import Dataset
import os
import posixpath
import numpy as np
import torch
# TODO: implement here your custom dataset class for Cityscapes
from torchvision.io import read_image 
from torchvision import transforms
from torchvision.transforms import v2
from torchvision import tv_tensors
import torchvision.transforms.functional as F
def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image
#custom transformation for padding. Given a target and a fill, it applies a costant padding.
#Useful when image is cropped and it's necessary to feed an images with constant size.
class SquarePad:
  def __call__(self, image,target,fill=0):
    s = image.shape
    h=target[-2]
    w=target[-1]
    hp = int((h - s[-2])/2)
    vp = int((w - s[-1])/2)
    padding = (vp, hp, vp, hp)
    return F.pad(image, padding, fill, 'constant')
class CityScapes(Dataset):
    def __init__(self,annotations_dir,images_dir,transform=None, target_transform=None,applier=None):
        super(CityScapes, self).__init__()
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.target_transform = target_transform
        self.map_index_to_image = []
        self.map_index_to_annotation = []
        self.generate_map_images(path=self.images_dir)
        self.generate_map_annotations(path=self.annotations_dir)
        self.pad_transformation = SquarePad()
        # set of transformation that are applied during training for data augmentation purposes
        self.applier = applier
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
            #else:
            elif full_path.endswith("Ids.png"):
                self.map_index_to_annotation.append(full_path)

    def __getitem__(self, idx):
        image=""
        annotation=""
        image_path = self.map_index_to_image[idx]
        image = read_image(image_path)
        #return last two elements of the path
        path=image_path.split('/')
        image_name = posixpath.join(path[-2],path[-1])
        
        annotation_path = posixpath.join(self.annotations_dir, image_name.replace("_leftImg8bit.png","_gtFine_labelTrainIds.png"))
        #annotation_path = self.map_index_to_annotation[idx]
        annotation = read_image(annotation_path)#[0:3,:,:]
        #convert annotation to mask tv_tensor; In this way, transforms are able to properly transform the label.
        annotation_tv=tv_tensors.Mask(annotation)
        #image_t,annotation_t=applier2(image,annotation_tv)
        if self.applier:
            image_shape=image.shape
            annotation_shape=annotation.shape
            image,annotation=self.applier(image,annotation_tv)
            #check if cropping has been executed. In that case, apply padding 
            if image.shape != image_shape:
                image=self.pad_transformation(image,image_shape)
                annotation=self.pad_transformation(annotation,annotation_shape,fill=255)
            #v2.RandomApply(transforms=[v2.RandomCrop(size=(256, 256))], p=1)
        if self.transform:
             image = self.transform(image)
             annotation= torch.tensor(self.transform(annotation),dtype=torch.uint8)
        return image, annotation

    def __len__(self):
        return len(self.map_index_to_image)
 
