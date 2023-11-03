import pandas as pd
import os

from skimage.exposure import equalize_adapthist, equalize_hist
from skimage.filters import threshold_otsu, threshold_triangle

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


class CellDataset(Dataset):
    def __init__(self, meta_data, image_data, transforms=None, uri_field="uri"):
        """
        Args:
        """
        meta_file = os.path.join(meta_data, 'train.csv')
        df = pd.read_csv(meta_file)
        df = df.loc[df['modality_name'] == 'BrightField']
        df['uri'] = df.uri.str.replace('file:///images', image_root)#'s3://syedazi-demo-content-aiml-team/bayer-crop-science-poc/fungi-cell/')

        self.uri_field = uri_field
        self.transforms = transforms
        self.meta_data = df

    def get_image(self, row: pd.Series):
        url = row[self.uri_field]
        image = np.array(Image.open(url)).astype(np.uint16)
        return image
        
    def __len__(self):
        return self.meta_data.index.size

    def __getitem__(self, idx):

        meta_data = self.meta_data.iloc[idx]

        image = self.get_image(meta_data)

        image = equalize_adapthist(image)
        image = rescale_intensity(image, q=1e-2)
        thresh = threshold_otsu(image)
        # Convert binary image back to uint8

        binary = image > thresh

        # Normalize between 0-1 using PIL
        image = Image.fromarray(binary).convert('L')
        image = np.array(image) / 255.0
        #image = image / image.max()
        if self.transforms:
            image = self.transforms(image=image)

        return image, meta_data
    
    def collate(batch):
        import numpy as np
        import pandas as pd
        import torch

        img = []
        meta = []
        stats = []
        for s in batch:
            im = s["image"]
            img.append(im)
            meta.append(s["meta_data"])

        img = torch.from_numpy(np.array(img)).float().unsqueeze(1)

        return img, pd.DataFrame(meta)

