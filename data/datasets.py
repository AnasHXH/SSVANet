# data/datasets.py

import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class ShadowRemovalDataset(Dataset):
    def __init__(self, gt_dir, lq_dir, transform=None, augment=False):
        self.gt_dir = gt_dir
        self.lq_dir = lq_dir
        self.transform = transform
        self.augment = augment
        self.gt_images = sorted(os.listdir(gt_dir))
        self.lq_images = sorted(os.listdir(lq_dir))

    def __len__(self):
        return len(self.gt_images)

    def augment_images(self, gt_image, lq_image):
        # Random rotation
        angle = random.choice([0, 90, 180, 270])
        gt_image = TF.rotate(gt_image, angle)
        lq_image = TF.rotate(lq_image, angle)

        # Random horizontal flipping
        if random.random() > 0.25:
            gt_image = TF.hflip(gt_image)
            lq_image = TF.hflip(lq_image)

        # Random vertical flipping
        if random.random() > 0.25:
            gt_image = TF.vflip(gt_image)
            lq_image = TF.vflip(lq_image)

        return gt_image, lq_image

    def __getitem__(self, idx):
        gt_img_path = os.path.join(self.gt_dir, self.gt_images[idx])
        lq_img_path = os.path.join(self.lq_dir, self.lq_images[idx])

        gt_image = Image.open(gt_img_path).convert("RGB")
        lq_image = Image.open(lq_img_path).convert("RGB")

        if self.augment:
            gt_image, lq_image = self.augment_images(gt_image, lq_image)

        if self.transform:
            gt_image = self.transform(gt_image)
            lq_image = self.transform(lq_image)

        return lq_image, gt_image
