import multiprocessing
from copy import deepcopy

import pandas as pd
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.nn import Module
from torch.utils.data import DataLoader

from datasets.pneumonia_dataset import PneumoniaDataset

# Note - you must have torchvision installed for this example


class PneumoniaDataModule(LightningDataModule):
    """Pneumonia data module"""

    def __init__(
        self,
        data: pd.DataFrame,
        required_transforms: Module,
        test_size: float = 0.1,
        label_col: str = "pneumonia",
        batch_size: int = 32,
        random_seed: bool = 8080,
    ):
        """
        Data module init

        Args:
            data: Dataframe containing patient metadata and path to xrax image.
            required_transforms: Must-have transforms for EfficientNetV2.
            test_size: Percentage of data destined for the testing set.
            label_col: Column containing target labels. Also used for stratified splits.
            batch_size: How many images to load in a single batch.
            random_seed: Seed to initialize random state.
        """
        super().__init__()
        self.data = data
        self.test_size = test_size
        self.label_col = label_col
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.pre_processing = T.Compose([required_transforms])
        self.data_augmentation = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            ]
        )

    def setup(self, stage: str):
        # 1. Train/test split
        train_data, test_data = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.random_seed,
            shuffle=True,
            stratify=self.data[self.label_col],
        )

        # 2. Train/val split
        train_data, val_data = train_test_split(
            train_data,
            test_size=self.test_size,
            random_state=self.random_seed,
            shuffle=True,
            stratify=train_data[self.label_col],
        )

        # 3. Setup datasets
        train_transforms = deepcopy(self.pre_processing)
        train_transforms.transforms.extend(self.data_augmentation.transforms)
        self.train_dataset = PneumoniaDataset(train_data, train_transforms)
        self.val_dataset = PneumoniaDataset(val_data, self.pre_processing)
        self.test_dataset = PneumoniaDataset(test_data, self.pre_processing)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
        )
