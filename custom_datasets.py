import os
from PIL import Image
import torch
from torch import random
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        print(transform)
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) # greyscale
        mask /= 255.0

        if self.transform is not None:
            state = random.get_rng_state()
            seed = torch.randint(0, 2**32, ())
            random.manual_seed(seed)
            image = self.transform(image)
            random.manual_seed(seed)
            mask = self.transform(mask)
            random.set_rng_state(state)
        
        return image, mask