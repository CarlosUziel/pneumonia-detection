from typing import Callable

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


class PneumoniaDataset(Dataset):
    """A dataset containing x-rax images with the presence of absence of pneumonia"""

    def __init__(self, data: pd.DataFrame, transform: Callable = None):
        """
        Args:
            data: Dataframe containing patient metadata and path to xrax image.
            transform: Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx.tolist() if isinstance(idx, Tensor) else idx

        image = read_image(
            str(self.data.iloc[idx]["image_path"]), mode=ImageReadMode.GRAY
        ) * (1.0 / 255)
        target = torch.tensor(float(self.data.iloc[idx]["pneumonia"]))

        if self.transform:
            self.transform(image)

        return image, target
