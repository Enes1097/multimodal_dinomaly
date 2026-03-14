from lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Sequence, Optional

import random
from pathlib import Path
from torchvision.transforms.v2 import Transform
from anomalib.data.datasets.image.multimodalad import MultiModalFolderDataset

from torch.utils.data import Dataset, Subset, WeightedRandomSampler

import torch

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
        balance_object_classes: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        # required by anomalib.Engine
        self.name: str = self.root.name if self.root is not None else "multimodal" #Sets the dataset name, e.g., "yellow_cup"
        self.category: str = "image"
        self.modalities = list(modalities) #Make list from sequence
        self.extensions = tuple(e.lower() for e in extensions) #Same as with dataset only lower case extensions are allowed
        self.train_ratio_normal = train_ratio_normal
        self.val_ratio_normal = val_ratio_normal
        self.val_ratio_anom = val_ratio_anom
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.transform = transform
        self.balance_object_classes = balance_object_classes

        self.train_data: Dataset | None = None #empty datasets
        self.val_data: Dataset | None = None
        self.test_data: Dataset | None = None

    @staticmethod
    def _infer_object_class_from_path(path: Path) -> str | None:
        """Infer object class from a multiclass curated path.

        Expected layout (as created by scripts/datasets/create_multiclass_dataset.py):
            <root>/<modality>/<normal|anomalous>/<object_class>/.../<file>
        """
        parts = path.parts
        for label_name in ("normal", "anomalous"):
            try:
                label_index = parts.index(label_name)
            except ValueError:
                continue
            if label_index + 1 < len(parts):
                return parts[label_index + 1]
        return None

    def setup(self, stage: str | None = None) -> None: #Builds the dataset with splits
        # 1) Vollständiges Dataset ohne Splits.
        full_dataset = MultiModalFolderDataset(
            root=self.root,
            modalities=self.modalities,
            extensions=self.extensions,
            transform=self.transform,
        )

        labels = [s["label"] for s in full_dataset.samples] #List of 0s and 1s for all samples
        normal_indices = [i for i, y in enumerate(labels) if y == 0] #List of normal indices storing at which list index a normal sample is located
        anom_indices = [i for i, y in enumerate(labels) if y == 1] #List of anomalous indices storing at which list index an anomalous sample is located

        rng = random.Random(self.seed)
        rng.shuffle(normal_indices) #Shuffle indices with seed
        rng.shuffle(anom_indices) #Shuffle indices with seed

        # --- Normale Splits ---
        N = len(normal_indices)
        tr = self.train_ratio_normal
        vr = self.val_ratio_normal
        if tr + vr > 1.0:
            raise ValueError("train_ratio_normal + val_ratio_normal darf nicht > 1.0 sein")
        n_train = int(N * tr)
        n_val = int(N * vr)

        train_norm_idx = normal_indices[:n_train]
        val_norm_idx = normal_indices[n_train:n_train + n_val]
        test_norm_idx = normal_indices[n_train + n_val:]

        # --- Anomalie-Splits (nur val/test) ---
        M = len(anom_indices)
        va = self.val_ratio_anom
        if va > 1.0:
            raise ValueError("val_ratio_anom darf nicht > 1.0 sein")
        n_val_anom = int(M * va)

        val_anom_idx = anom_indices[:n_val_anom]
        test_anom_idx = anom_indices[n_val_anom:]

        train_indices = train_norm_idx
        val_indices = val_norm_idx + val_anom_idx
        test_indices = test_norm_idx + test_anom_idx

        val_labels = [full_dataset.samples[i]["label"] for i in val_indices]
        test_labels = [full_dataset.samples[i]["label"] for i in test_indices]

        print("[SPLIT]")
        print(f"  normals: total={N}, train={len(train_norm_idx)}, val={len(val_norm_idx)}, test={len(test_norm_idx)}")
        print(f"  anoms  : total={M}, val={len(val_anom_idx)}, test={len(test_anom_idx)}")
        print(f"  final  : train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        print("[SANITY]")
        print(
            "  val label counts : "
            f"normal={sum(y == 0 for y in val_labels)}, anomalous={sum(y == 1 for y in val_labels)}"
        )
        print(
            "  test label counts: "
            f"normal={sum(y == 0 for y in test_labels)}, anomalous={sum(y == 1 for y in test_labels)}"
        )

        if test_labels:
            assert any(y == 0 for y in test_labels), "Test split has no normal samples."
            assert any(y == 1 for y in test_labels), "Test split has no anomalous samples."

        self.train_data = Subset(full_dataset, train_indices) #Subsets refer to the full dataset so data is not copied
        self.val_data = Subset(full_dataset, val_indices)
        self.test_data = Subset(full_dataset, test_indices)

    # Initialize dataloaders
    def train_dataloader(self) -> DataLoader:
        assert self.train_data is not None
        if self.train_data is None or len(self.train_data) == 0:
            raise ValueError("train_data is not set or empty. Call setup() or adjust split ratios.")
        # Initiate over-sampling of object classes if desired and if possible
        sampler = None
        shuffle = True

        if self.balance_object_classes:
            if not isinstance(self.train_data, Subset):
                raise TypeError("Expected train_data to be a Subset when balance_object_classes is enabled.")

            base_dataset = self.train_data.dataset
            if not hasattr(base_dataset, "samples"):
                raise AttributeError(
                    "balance_object_classes requires the underlying dataset to expose a 'samples' attribute."
                )

            object_classes: list[str] = []
            for base_index in self.train_data.indices:
                sample = base_dataset.samples[base_index]
                paths = sample.get("paths", {})
                ref_path = paths.get("thermal") or next(iter(paths.values()))
                inferred = self._infer_object_class_from_path(Path(ref_path))
                object_classes.append(inferred or "__unknown__")

            counts: dict[str, int] = {}
            for name in object_classes:
                counts[name] = counts.get(name, 0) + 1

            weights = torch.tensor([1.0 / counts[name] for name in object_classes], dtype=torch.double)
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
            shuffle = False

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
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