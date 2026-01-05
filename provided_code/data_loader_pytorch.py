from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch

from monai.transforms import Compose
from provided_code.utils import get_paths, load_file
from provided_code.data_shapes import DataShapes

from monai.data import PersistentDataset, ThreadDataLoader
from pathlib import Path
from typing import List, Optional
import lightning as L
from monai.transforms import Compose

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from typing import List, Optional, Callable
from monai.transforms import Compose
import lightning as L

class OpenKBPDataset(Dataset):
    def __init__(
        self,
        patient_paths: List[Path],
        mode: str = "training_model",
        transforms: Optional[Compose] = None,
        rois: Optional[Dict[str, List[str]]] = None,
    ):
        self.patient_paths = patient_paths
        self.mode = mode
        self.transforms = transforms

        # ROI and data shape setup
        self.rois = rois or {
            "oars": ["Brainstem", "SpinalCord", "RightParotid", "LeftParotid", "Esophagus", "Larynx", "Mandible"],
            "targets": ["PTV56", "PTV63", "PTV70"],
        }
        self.full_roi_list = sum(self.rois.values(), [])
        self.num_rois = len(self.full_roi_list)
        self.data_shapes = DataShapes(self.num_rois)
        self.required_files = self._get_required_files()

    def _get_required_files(self) -> Dict[str, tuple]:
        if self.mode == "training_model":
            keys = ["dose", "ct", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
        elif self.mode == "predicted_dose":
            keys = [self.mode]
        elif self.mode == "evaluation":
            keys = ["dose", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
        elif self.mode == "dose_prediction":
            keys = ["ct", "structure_masks", "possible_dose_mask", "voxel_dimensions"]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return self.data_shapes.from_data_names(keys)

    def __len__(self):
        return len(self.patient_paths)

    def __getitem__(self, idx):
        patient_path = self.patient_paths[idx]
        data = self.load_data(patient_path)

        sample = {}
        for key in self.required_files:
            sample[key] = self.shape_data(key, data)

        sample["patient_id"] = patient_path.stem

        if self.transforms:
            sample = self.transforms(sample)

        # Convert arrays to torch tensors
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                sample[k] = torch.from_numpy(v).float()

        return sample

    def load_data(self, path: Path) -> Dict[str, np.ndarray]:
        data = {}
        if path.is_dir():
            for file in get_paths(path):
                if file.stem in self.required_files or file.stem in self.full_roi_list:
                    data[file.stem] = load_file(file)
        else:
            data[self.mode] = load_file(path)
        return data

    def shape_data(self, key: str, data: Dict[str, np.ndarray]) -> np.ndarray:
        shaped = np.zeros(self.required_files[key], dtype=np.float32)
        if key == "structure_masks":
            for i, roi in enumerate(self.full_roi_list):
                if roi in data:
                    np.put(shaped, self.num_rois * data[roi] + i, 1)
        elif key == "possible_dose_mask":
            np.put(shaped, data[key], 1)
        elif key == "voxel_dimensions":
            shaped = data[key]
        else:
            np.put(shaped, data[key]["indices"], data[key]["data"])
        return shaped
    



class OpenKBPDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_paths: List[Path],
        val_paths: List[Path],
        test_paths: Optional[List[Path]] = None,
        train_transforms: Optional[Compose] = None,
        val_transforms: Optional[Compose] = None,
        test_transforms: Optional[Compose] = None,
        batch_size: int = 2,
        num_workers: int = 4,
        mode: str = "training_model",
        cache_dir: Optional[Path] = Path("/tmp/openkbp_cache"),  # ✅ new
    ):
        super().__init__()
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.cache_dir = Path(cache_dir)

    def setup(self, stage: Optional[str] = None):
        # Convert Paths to MONAI dict-style samples
        train_data = [{"patient_path": p} for p in self.train_paths]
        val_data = [{"patient_path": p} for p in self.val_paths]
        test_data = [{"patient_path": p} for p in self.test_paths] if self.test_paths else None

        # Custom loader to wrap the OpenKBPDataset
        def dataset_wrapper(data_list, transforms):
            return PersistentDataset(
                data=data_list,
                transform=lambda sample: OpenKBPDataset([sample["patient_path"]], self.mode, transforms)[0],
                cache_dir=self.cache_dir,
            )

        self.train_dataset = dataset_wrapper(train_data, self.train_transforms)
        self.val_dataset = dataset_wrapper(val_data, self.val_transforms)
        if test_data:
            self.test_dataset = dataset_wrapper(test_data, self.test_transforms)

    def train_dataloader(self):
        return ThreadDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return ThreadDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_paths:
            return ThreadDataLoader(
                self.test_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )