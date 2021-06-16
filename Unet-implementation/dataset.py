import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class CarvanaDataset(Dataset):
    def __init__(self, image_dir=None, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.images = np.array(os.listdir(self.image_dir))
        self.masks = np.array(os.listdir(self.mask_dir))

        sort_index = np.argsort(self.images)
        self.images = self.images[sort_index]
        self.masks = self.masks[sort_index]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])

        image = np.asarray(Image.open(img_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def make_dataloaders(
    batch_size=32, n_workers=4, pin_memory=False, shuffle=True, **kwargs
):  # A handy function to make our dataloaders
    dataset = CarvanaDataset(**kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
    )
    return dataloader

