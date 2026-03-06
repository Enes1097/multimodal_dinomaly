from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Sequence, Optional

import random
from pathlib import Path
from torchvision.transforms.v2 import Transform
from anomalib.data.datasets.image.multimodalad import MultiModalFolderDataset

from torch.utils.data import Dataset, Subset

class MultiModalDataModule(LightningDataModule):
    """DataModule mit train/val/test Split für MultiModalFolderDataset.

    - root: unsplitteter multimodaler Ordner (normal/anomalous/...).
    - train nutzt NUR normale Samples.
    - val/test nutzen normal + anomalous.

    Splits:
        normals:
            train_ratio_normal
            val_ratio_normal
            test_normal = 1 - train_ratio_normal - val_ratio_normal

        anomalous:
            val_ratio_anom
            test_anom = 1 - val_ratio_anom
    """

    def __init__(
        self,
        root: str | Path,
        modalities: Sequence[str] = ("thermal", "rgb"),
        extensions: Sequence[str] = (".png", ".jpg", ".jpeg"),
        train_ratio_normal: float = 0.7,
        val_ratio_normal: float = 0.15,
        val_ratio_anom: float = 0.5,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 42,
        transform: Optional[Transform] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        # required by anomalib.Engine
        self.name: str = self.root.name if self.root is not None else "multimodal"
        self.category: str = "image"
        self.modalities = list(modalities)
        self.extensions = tuple(e.lower() for e in extensions)
        self.train_ratio_normal = train_ratio_normal
        self.val_ratio_normal = val_ratio_normal
        self.val_ratio_anom = val_ratio_anom
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.transform = transform

        self.train_data: Dataset | None = None
        self.val_data: Dataset | None = None
        self.test_data: Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        # 1) Vollständiges Dataset ohne Splits.
        full_dataset = MultiModalFolderDataset(
            root=self.root,
            modalities=self.modalities,
            extensions=self.extensions,
            transform=self.transform,
        )

        labels = [s["label"] for s in full_dataset.samples]
        normal_indices = [i for i, y in enumerate(labels) if y == 0]
        anom_indices = [i for i, y in enumerate(labels) if y == 1]

        rng = random.Random(self.seed)
        rng.shuffle(normal_indices)
        rng.shuffle(anom_indices)

        # --- Normale Splits ---
        N = len(normal_indices)
        tr = self.train_ratio_normal
        vr = self.val_ratio_normal
        if tr + vr > 1.0:
            raise ValueError("train_ratio_normal + val_ratio_normal darf nicht > 1.0 sein")
        n_train = int(N * tr)
        n_val = int(N * vr)
        #n_test = N - n_train - n_val

        train_norm_idx = normal_indices[:n_train]
        val_norm_idx = normal_indices[n_train:n_train + n_val]
        test_norm_idx = normal_indices[n_train + n_val:]

        # --- Anomalie-Splits (nur val/test) ---
        M = len(anom_indices)
        va = self.val_ratio_anom
        if va > 1.0:
            raise ValueError("val_ratio_anom darf nicht > 1.0 sein")
        n_val_anom = int(M * va)
        #n_test_anom = M - n_val_anom

        val_anom_idx = anom_indices[:n_val_anom]
        test_anom_idx = anom_indices[n_val_anom:]

        train_indices = train_norm_idx
        val_indices = val_norm_idx + val_anom_idx
        test_indices = test_norm_idx + test_anom_idx

        print("[SPLIT]")
        print(f"  normals: total={N}, train={len(train_norm_idx)}, val={len(val_norm_idx)}, test={len(test_norm_idx)}")
        print(f"  anoms  : total={M}, val={len(val_anom_idx)}, test={len(test_anom_idx)}")
        print(f"  final  : train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

        self.train_data = Subset(full_dataset, train_indices)
        self.val_data = Subset(full_dataset, val_indices)
        self.test_data = Subset(full_dataset, test_indices)

    def train_dataloader(self) -> DataLoader:
        assert self.train_data is not None
        if self.train_data is None or len(self.train_data) == 0:
            raise ValueError("train_data is not set or empty. Call setup() or adjust split ratios.")
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_data is not None
        if self.val_data is None or len(self.val_data) == 0:
            raise ValueError("val_data is not set or empty. Call setup() or adjust split ratios.")
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_data is not None
        if self.test_data is None or len(self.test_data) == 0:
            raise ValueError("test_data is not set or empty. Call setup() or adjust split ratios.")
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )