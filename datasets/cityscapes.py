from torch.utils.data import Dataset
import os
import posixpath
import numpy as np
# TODO: implement here your custom dataset class for Cityscapes
from torchvision.io import read_image 
def denormalize(image):
    image = image.to('cpu').numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image
class CityScapes(Dataset):
    def __init__(self,annotations_dir,images_dir,transform=None, target_transform=None):
        super(CityScapes, self).__init__()
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.target_transform = target_transform
        self.map_index_to_image = {}
        self.map_index_to_annotation = {}
        
        counter=self.generate_map_images(path=self.images_dir)
        #print(counter)
        counter=self.generate_map_annotations(path=self.annotations_dir)
        #print(len(self.map_index_to_annotation))
        #print(counter)

    """def list_files_recursive(self,path='.',counter=0):
        for entry in os.listdir(path):
            full_path = posixpath.join(path, entry)
            if os.path.isdir(full_path):
                counter=counter+self.list_files_recursive(full_path)
            else:
                counter = counter+1
        return counter"""
    def generate_map_images(self,path='.',counter=0):
        
        for entry in os.listdir(path):
            full_path = posixpath.join(path, entry)
            if os.path.isdir(full_path):
                counter=self.generate_map_images(full_path,counter)
            else:

                self.map_index_to_image[counter]=full_path
                counter = counter+1

        return counter
    def generate_map_annotations(self,path='.',counter=0):
        for entry in os.listdir(path):
            full_path = posixpath.join(path, entry)
            if os.path.isdir(full_path):
                counter=self.generate_map_annotations(full_path,counter)
            elif full_path.endswith("color.png"):
                self.map_index_to_annotation[counter]=full_path
                counter = counter+1
        return counter

    def __getitem__(self, idx):
        image=""
        annotation=""
        image_path = self.map_index_to_image[idx]
        annotation_path = self.map_index_to_annotation[idx]
        image = read_image(image_path).permute(1, 2, 0)
        annotation = read_image(annotation_path).permute(1,2,0)
        return image, annotation

    def __len__(self):
        return len(self.map_index_to_annotation)
 

    # def __getitem__(self, idx):
    #     img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    #     image = read_image(img_path)
    #     label = self.img_labels.iloc[idx, 1]
    #     if self.transform:
    #         image = self.transform(image)
    #     if self.target_transform:
    #         label = self.target_transform(label)
    #     return image, label
    
    #step 1: map each index to an image
    #step 2: map each index to a label