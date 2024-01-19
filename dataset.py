import os
import cv2 as cv
import numpy as np
import torch 

from random import random
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from torchvision.transforms import functional as tf

from typing import Optional, Callable

def train_transform(image, mask, img_size):
  image = torch.from_numpy(image).float().unsqueeze(0)
  mask = torch.from_numpy(mask).float().unsqueeze(0)

  # this is done to maintain same transforms between image and mask      
  # random resized crop
  rrc = transforms.RandomResizedCrop(size=img_size, antialias=None)
  rrc_params = rrc.get_params(image, scale=(0.08, 1.0), ratio=(0.75, 1.33333333333))

  image = tf.resized_crop(image, *rrc_params, size=img_size)
  mask = tf.resized_crop(mask, *rrc_params, size=img_size)
    
  # print(image.shape, mask.shape)

  # random horizontal flip
  if random() > 0.5:
    image = tf.hflip(image)
    mask = tf.hflip(mask)

  # normalize, this is wrong but why
  # image = tf.normalize(image, mean=(0.5,), std=(0.5,))
  # mask = tf.normalize(mask, mean=(0.5,), std=(0.5,))
      
  # scale instead of normalize bcus we cannot normalize white masks
  image /= 255
  mask /= 255
    
  return image, mask

def eval_transforms(image, mask, img_size):
  # print()
  # print()
  # print(image[0].shape, image[1].shape)
  # print(type(mask))
  # print()
  # print()
  image = torch.from_numpy(image).float().unsqueeze(0)
  mask = torch.from_numpy(mask).float().unsqueeze(0)

  image = tf.resize(image, img_size)
  mask = tf.resize(mask, img_size)

  # # normalize
  # image = tf.normalize(image, mean=0.5, std=0.5)
  # mask = tf.normalize(mask, mean=0.5, std=0.5)

  # scale instead of normalize bcus we cannot normalize white masks
  image /= 255
  mask /= 255

  return image, mask

class KMT(Dataset):
  def __init__(
      self, 
      data_dir: Path, 
      transforms: str = None,
      train: bool = False, 
      crop_size: tuple = None
    ):
    self.data_dir = data_dir
    self.train = train
    self.crop_size = crop_size
    self.transforms = transforms
    self.current_set = self.get_current_set()
    self.images, self.masks, self.labels = self.fill_dataset()

  def __len__(self):
    total_images, total_masks = len(self.images), len(self.masks)
    assert total_images == total_masks, "Total length of images and masks do not match."
    return total_images

  def __getitem__(self, idx):
    image, mask, label = self.images[idx], self.masks[idx], self.labels[idx]
    return image, mask, label

  def get_current_set(self):
    set_dir = self.data_dir / "train" if self.train else self.data_dir / "test"
    return set_dir

  def _get_defect_types(self):
    return os.listdir(self.current_set)

  def _get_image_n_mask_dir(self, defect_dir):
    return defect_dir / "images", defect_dir / "masks"

  def _get_data(self, defect_type):
    defect_dir = self.current_set / defect_type

    is_good_sample = defect_type == "Good"
      
    if is_good_sample:
      image_dir = defect_dir / "images"
      images = np.array([cv.imread(str(image_dir / f), cv.IMREAD_GRAYSCALE) for f in os.listdir(image_dir)])
      masks = np.full_like(images, 255, dtype=np.uint8)
      return images, masks
    
    else:
      image_dir, mask_dir = self._get_image_n_mask_dir(defect_dir)
      images = np.array([cv.imread(str(image_dir / f), cv.IMREAD_GRAYSCALE) for f in os.listdir(image_dir)])
      masks = np.array([cv.imread(str(mask_dir / f), cv.IMREAD_GRAYSCALE) for f in os.listdir(mask_dir)])
      return images, masks

  def fill_dataset(self):
    images, masks, labels = [], [], []

    for d in self._get_defect_types():
      imgs, msks = self._get_data(d)
      images.append(imgs)
      masks.append(msks)
      labels.append([d for _ in range(len(imgs))])

    images = np.concatenate(images, axis=0)
    masks = np.concatenate(masks, axis=0)
    assert images.shape[0] == masks.shape[0]
    labels = np.concatenate(labels, axis=0)
    
    le = LabelEncoder()
    le.fit(labels)

    self.label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    encoded_labels = np.array([self.label_mapping[lbl] for lbl in labels])

    assert images.shape[0] == masks.shape[0] == encoded_labels.shape[0]
    return images, masks, encoded_labels

class Subset(Dataset):
  def __init__(
      self, 
      subset, 
      crop_size: tuple = None,
      transforms: str = None
    ):
    self.subset = subset
    self.crop_size = crop_size
    self.transforms = transforms

  def __len__(self):
    return len(self.subset)    

  def __getitem__(self, index):
    image, mask, label = self.subset[index]

    # this is done to maintain same transforms between image and mask      
    if self.transforms == "train_transforms":
      d_i, m_i = train_transform(image, mask, self.crop_size)
      d_j, m_j = train_transform(image, mask, self.crop_size)
    elif self.transforms == "eval_transforms":
      d_i, m_i = eval_transforms(image, mask, self.crop_size)
      d_j, m_j = train_transform(image, mask, self.crop_size)
    else:
      raise ValueError("Transforms unavailable.")

    return (d_i, m_i), (d_j, m_j), label, index
          
class KMTDataModule(LightningDataModule):
  def __init__(
    self, 
    data_dir: Path = Path("./KMT-data/"), 
    crop_size: tuple = (512, 512), 
    batch_size: int = 500, 
    train_transforms: str = 'train_transforms', 
    coreset_select=False
  ):
    super().__init__()
    self.save_hyperparameters()
      
  def setup(self, stage=None):
    self.kmt_train = KMT(
      self.hparams.data_dir, 
      train=True, 
      crop_size=self.hparams.crop_size, 
      transforms="train_transforms"
    )

    gen = torch.Generator().manual_seed(42)
    train, val = random_split(self.kmt_train, [0.7, 0.3], generator=gen)
    self.kmt_train = Subset(train, self.hparams.crop_size, "train_transforms")
    self.kmt_val = Subset(val, self.hparams.crop_size, "eval_transforms")
    
  def train_dataloader(self):
    return DataLoader(self.kmt_train, self.hparams.batch_size)

  def val_dataloader(self):
    return DataLoader(self.kmt_val, self.hparams.batch_size)

  def test_dataloader(self):
    return DataLoader()

  def get_coreset(self):
    if self.hparams.coreset_select:
      # read final averaged cossim
      cossim_avg = np.load('./cossim_outputs/cossim_avg.npy')

      # read cossim_history (for analysis)
      cossim_hist = np.load('./cossim_outputs/cossim_history.npy')

      # sort by least to most cossim score
      sorted_indices = np.argsort(cossim_avg)

      loader = DataLoader(torch.utils.data.Subset(self.kmt_train, indices=sorted_indices))

      return loader, cossim_avg, cossim_hist, sorted_indices
    else:
      raise ValueError(f"coreset_select parameter is set to {self.hparams.coreset_select}. Change to True.")