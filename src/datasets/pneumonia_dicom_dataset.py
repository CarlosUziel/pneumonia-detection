from copy import deepcopy
from typing import Callable

import pandas as pd
import torch
import torchvision.transforms as T
from pydicom import dcmread
from torch import Tensor
from torch.utils.data import Dataset


class PneumoniaDicomDataset(Dataset):
    """A dataset containing DICOM files with the presence or absence of pneumonia"""

    def __init__(self, data: pd.DataFrame, transform: Callable = None):
        """
        Args:
            data: Dataframe containing patient metadata and path to dicom file.
            transform: Optional transform to be applied
                on a sample.
        """
        # filter out invalid DICOM files
        self.data = deepcopy(
            data[
                (data["modality"] == "DX")
                & (data["body_part_examined"] == "CHEST")
                & (data["patient_position"].isin(("AP", "PA")))
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx.tolist() if isinstance(idx, Tensor) else idx

        image = T.ToTensor()(
            dcmread(str(self.data.iloc[idx]["file_path"])).pixel_array
        ).repeat(3, 1, 1)

        target = torch.tensor(
            float(self.data.iloc[idx]["study_description"] == "Pneumonia")
        )

        if self.transform:
            self.transform(image)

        return image, target
