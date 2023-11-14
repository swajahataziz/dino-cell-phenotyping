import torch
from torchvision import datasets
import utils
from tifffile import imread
import numpy
import torchvision.transforms as transforms
import json
import yaml
import os

import pandas as pd
from torch.utils.data import Dataset
import tifffile
import numpy as np

def save_config_file(config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/run_config_dump.json", "w") as f:
        json.dump(config, f)
    with open(f"{save_dir}/run_config_dump.yaml", "w") as f:
        yaml.dump(config, f)


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

config_file = 'DINO_full_pipeline.yaml'
with open(config_file, "r") as f:
    config = yaml.load(f,Loader=yaml.FullLoader)

data_dir = config['meta']['dataset_dir'] 
fraction_for_mean_std = config['train_DINO']['fraction_for_mean_std_calc']
center_crop = config['meta']['center_crop']

name_of_run = config['meta']['name_of_run']
sk_save_dir = config['meta']['output_dir']
save_dir_downstream_run = sk_save_dir+"/"+name_of_run

output_file = f"{save_dir_downstream_run}/mean_and_std_of_dataset.txt"

save_config_file(config, save_dir_downstream_run)

print("loading dataset...")
image_root=os.environ["SM_CHANNEL_TRAIN"]
local_root=os.environ["SM_CHANNEL_META"]

dataset_total = ReturnStatsDataset(local_root, image_root, uri_field='uri')
print("length of dataset: ", len(dataset_total))

#SAMPLER SECTION
validation_split = 1-float(fraction_for_mean_std)
shuffle_dataset = True
random_seed= 42
dataset_size = len(dataset_total)
indices = list(range(dataset_size))
split = int(numpy.floor(validation_split * dataset_size))
if shuffle_dataset :
    numpy.random.seed(random_seed)
    numpy.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

print(len(train_indices), len(val_indices))

print(f"Train dataset consists of {len(train_indices)} images.")

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)

image_data_loader = torch.utils.data.DataLoader(
    dataset_total,
    sampler=train_sampler,
    batch_size=int(len(train_indices)/10),
    num_workers=0, 
    #collate_fn=dataset_total.collate_fn, 
    drop_last=True)

print("Successfully loaded data.")

def batch_mean_and_sd(loader):
    cnt = 0
    picture, _ = next(iter(image_data_loader))
    b, c, h, w = picture.shape
    fst_moment = torch.empty(c)
    snd_moment = torch.empty(c)
    
    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return mean,std
  
mean, std = batch_mean_and_sd(image_data_loader)
print("mean and std: \n", mean, std)


with open(output_file, 'w') as f:
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)
