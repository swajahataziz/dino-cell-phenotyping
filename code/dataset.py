import pandas as pd
import os

from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.transform import resize
import tifffile

import torch
from tqdm import tqdm
from imgaug import augmenters as iaa
import imgaug
import numpy as np
from urllib.parse import urlparse, urlunparse
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, RandomSampler
import boto3
import utils
from tifffile import imread

client = boto3.client('s3')
local_root=os.environ["SM_CHANNEL_META"]
image_root=os.environ["SM_CHANNEL_TRAIN"]

def rescale_intensity(img, q):

    import numpy as np
    import skimage as sk

    img = sk.exposure.rescale_intensity(
        img, in_range=tuple(np.quantile(img, q=(q, 1 - q)))
    )
    return img

OPTICAL_MAX_VALUE = 2000

class CellDataset(Dataset):
    def __init__(self, meta_data, image_data, transform=None, uri_field="uri"):
        """
        Args:
        """
        meta_file = os.path.join(meta_data, 'train.csv')
        df = pd.read_csv(meta_file)
        df = df.loc[df['modality_name'] == 'BrightField']
        df['uri'] = df.uri.str.replace('file:///images', image_root)#'s3://hpc-cluster-5d39e290/cell-data/images')

        self.uri_field = uri_field
        self.transform = transform
        self.meta_data = df

    def get_image(self, row: pd.Series):
        url = row[self.uri_field]
        # Open TIFF file
        tif = tifffile.TiffFile(url) 

        # Initialize empty array 
        image = np.empty((tif.pages[0].shape[0], 
                          tif.pages[0].shape[1],
                          len(tif.pages)), dtype=np.uint8)

        # Loop through TIFF pages (channels) and insert into image array
        for i, page in enumerate(tif.pages):
            image[:, :, i] = page.asarray()

        return image
        
    def __len__(self):
        return self.meta_data.index.size

    def __getitem__(self, idx):

        meta_data = self.meta_data.iloc[idx]

        image = self.get_image(meta_data)

        # apply image transforms
        image = torch.from_numpy(image).permute(2, 0, 1)
        #convert to 3 channel image to handle transformations
        image = image.repeat(3, 1, 1) # Convert to 3 channels
        image=np.array(image) 
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        return image, meta_data["cell_name"]
    
    def collate(self, batch):
        import numpy as np
        import pandas as pd
        import torch

        img = []
        meta = []
        stats = []
        for s in batch:
            im = s[0]
            img.append(im)
            meta.append(s[1])

        #img = torch.from_numpy(np.array(img)).float().unsqueeze(1)
        images = torch.stack(img) # stack into batch tensor
        print("Image stack size:"+str(images.size()))
        return images, pd.DataFrame(meta)

class ReturnStatsDataset(Dataset):
    def __init__(self, meta_data, image_data, uri_field="uri"):
        """
        Args:
        """
        meta_file = os.path.join(meta_data, 'train.csv')
        df = pd.read_csv(meta_file)
        df = df.loc[df['modality_name'] == 'BrightField']
        df['uri'] = df.uri.str.replace('file:///images', image_root)#'s3://hpc-cluster-5d39e290/cell-data/images')

        self.uri_field = uri_field
        self.meta_data = df

    def get_image(self, row: pd.Series):
        url = row[self.uri_field]
        # Open TIFF file
        tif = tifffile.TiffFile(url) 

        # Initialize empty array 
        image = np.empty((tif.pages[0].shape[0], 
                          tif.pages[0].shape[1],
                          len(tif.pages)), dtype=np.uint8)

        # Loop through TIFF pages (channels) and insert into image array
        for i, page in enumerate(tif.pages):
            image[:, :, i] = page.asarray()
        return image

    def __len__(self):
        return self.meta_data.index.size

    def __getitem__(self, idx):

        meta_data = self.meta_data.iloc[idx]
        image = self.get_image(meta_data)
        if center_crop != 0:
            image = torch.from_numpy(image).permute(2, 0, 1)
            transform = transforms.CenterCrop(center_crop)
            tensor = transform(image)
            image = image.permute(1, 2, 0)
            image = image.numpy()
        image = utils.normalize_numpy_0_to_1(image)
        if utils.check_nan(image):
            print("nan in image: ", path)
            return None
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1)
            if torch.isnan(tensor).any():
                print("nan in tensor: ", path)
                return None
            else:
                return tensor, idx

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

class MultichannelDataset(Dataset):
    def __init__(self, meta_data, image_data, channels, center_crop, transform=None, target_transform=None, uri_field="uri"):
        """
        Args:
        """
        meta_file = os.path.join(meta_data, 'train.csv')
        df = pd.read_csv(meta_file)
        df = df.loc[df['modality_name'] == 'BrightField']
        df['uri'] = df.uri.str.replace('file:///images', image_root)#'s3://hpc-cluster-5d39e290/cell-data/images')

        self.uri_field = uri_field
        self.transform = transform
        self.meta_data = df
        self.channels = channels
        self.center_crop = center_crop
        self.target_transform=target_transform

    def get_image(self, row: pd.Series):
        url = row[self.uri_field]

        return url

    def __len__(self):
        return self.meta_data.index.size

    def __getitem__(self, idx):
        meta_data = self.meta_data.iloc[idx]
        path = self.get_image(meta_data)
        #path, target = self.samples[idx]
        tif = tifffile.TiffFile(path)
        #image_np= imread(path)

        # Initialize empty array 
        image = np.empty((tif.pages[0].shape[0], 
                          tif.pages[0].shape[1],
                          len(tif.pages)), dtype=float)

        # Loop through TIFF pages (channels) and insert into image array
        for i, page in enumerate(tif.pages):
            image[:, :, i] = page.asarray()

        #image_np=image_np.astype(float)
        image_np = image[:,:,self.channels]
        if self.center_crop:
            image = torch.from_numpy(image_np).permute(2, 0, 1)
            transform = transforms.CenterCrop(self.center_crop)
            image = transform(image)
            image = image.permute(1, 2, 0)
            image_np = image.detach().cpu().numpy()
        image_np = utils.normalize_numpy_0_to_1(image_np)
        if utils.check_nan(image_np):
            print("nan in image: ", path)
            return None
        else:
            image = torch.from_numpy(image_np).float().permute(2, 0, 1)
            if self.transform is not None:
                image = self.transform(image)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return image, meta_data["cell_name"]
