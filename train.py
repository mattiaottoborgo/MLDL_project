# TODO: Define here your training and validation loops.
from datasets.cityscapes import CityScapes
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
dataset_path='C:/Users/Federico/Desktop/Universita/1_Magistrale/2_semestre/MLDL/proj/MLDL_project/datasets/Cityscapes/Cityscapes/Cityspaces/'
annotation_train=dataset_path+'gtFine/train'
image_train=dataset_path+'images/train'

annotation_val=dataset_path+'gtFine/val'
image_val=dataset_path+'images/val'

cityscapes_train = CityScapes(annotations_dir=annotation_train, images_dir=image_train)
cityscapes_val = CityScapes(annotations_dir=annotation_val, images_dir=image_val)
print(cityscapes_train.__len__())
print(cityscapes_val.__len__())
print(cityscapes_train.map_index_to_image[5])
print(cityscapes_train.map_index_to_annotation[5])
#cityscapes_train.list_files_recursive(path=image_train)
image,annotation=cityscapes_train.__getitem__(5)
fig, axes = plt.subplots(2, 1, figsize=(1, 10))
axes[0].imshow(image)
axes[1].imshow(annotation,cmap='gray')
plt.show()