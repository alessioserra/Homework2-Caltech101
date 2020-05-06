from torchvision.datasets import VisionDataset

from PIL import Image

import numpy as np
import os
import os.path
import sys
from sklearn.model_selection._split import StratifiedShuffleSplit


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split  # This defines the split you are going to use
                            # (split files are called 'train.txt' and 'test.txt')
        
        self.dataset = {}
        self.counter = 0
        classes_dict = {}
        class_counter = 0
        path = 'Caltech101/'+split+'.txt'
        indexes = set(np.loadtxt(path,dtype=str))
        classes = os.listdir(root)
        classes.remove('BACKGROUND_Google')
        
        for class_ in classes:
            classes_dict[class_] = class_counter
            class_counter += 1
            images = os.listdir(root+'/'+class_)
            for image in images:
                if class_+'/'+image in indexes:
                    self.dataset[self.counter] = (pil_loader(root+'/'+class_+'/'+image),classes_dict[class_])
                    self.counter += 1

    def __getitem__(self, index):
        image, label = self.dataset[index]
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.counter
    
    def _getsplit_(self, train_size):
        images, labels = [], []
        sss = StratifiedShuffleSplit(1,train_size=train_size)

        for item in self.dataset.values():
            images.append(item[0])
            labels.append(item[1])

        for x, y in sss.split(images,labels):
            train_indexes = x
            val_indexes = y 

        return train_indexes, val_indexes
