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
        anomaly_source_labels: Sequence[str] = ("anomalous",),
        mask_root: str | Path | None = None,
        mask_labels: Sequence[str] = (),
        require_masks_for_mask_labels: bool = True,
        skip_samples_without_masks_for_mask_labels: bool = False,
        restrict_normal_to_mask_classes: bool = False,
        train_ratio_normal: float = 0.7,
        val_ratio_normal: float = 0.15,
        val_ratio_anom: float = 0.5,
        batch_size: int = 16,
        num_workers: int = 8,
        seed: int = 42,
        transform: Optional[Transform] = None,
        balance_object_classes: bool = False,
        stratify_object_classes: bool = False,
        print_split_per_class: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        # required by anomalib.Engine
        self.name: str = self.root.name if self.root is not None else "multimodal" #Sets the dataset name, e.g., "yellow_cup"
        self.category: str = "image"
        self.modalities = list(modalities) #Make list from sequence
        self.extensions = tuple(e.lower() for e in extensions) #Same as with dataset only lower case extensions are allowed
        self.anomaly_source_labels = tuple(dict.fromkeys(anomaly_source_labels))
        self.mask_root = Path(mask_root) if mask_root is not None else None
        self.mask_labels = tuple(mask_labels)
        self.require_masks_for_mask_labels = require_masks_for_mask_labels
        self.skip_samples_without_masks_for_mask_labels = skip_samples_without_masks_for_mask_labels
        self.restrict_normal_to_mask_classes = restrict_normal_to_mask_classes
        self.train_ratio_normal = train_ratio_normal
        self.val_ratio_normal = val_ratio_normal
        self.val_ratio_anom = val_ratio_anom
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.transform = transform
        self.balance_object_classes = balance_object_classes
        self.stratify_object_classes = stratify_object_classes
        self.print_split_per_class = print_split_per_class

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
        for label_name in ("normal", "anomalous", "hotspot"):
            try:
                label_index = parts.index(label_name)
            except ValueError:
                continue
            if label_index + 1 < len(parts):
                return parts[label_index + 1]
        return None

    def _count_object_classes(self, dataset: MultiModalFolderDataset, indices: list[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for idx in indices:
            sample = dataset.samples[idx]
            paths = sample.get("paths", {})
            ref_path = paths.get("thermal") or next(iter(paths.values()))
            object_class = self._infer_object_class_from_path(Path(ref_path)) or "__unknown__"
            counts[object_class] = counts.get(object_class, 0) + 1
        return counts

    @staticmethod
    def _format_counts(counts: dict[str, int]) -> str:
        if not counts:
            return "(none)"
        parts = [f"{k}={counts[k]}" for k in sorted(counts.keys())]
        return ", ".join(parts)

    def setup(self, stage: str | None = None) -> None: #Builds the dataset with splits
        # 1) Vollständiges Dataset ohne Splits.
        full_dataset = MultiModalFolderDataset(
            root=self.root,
            modalities=self.modalities,
            extensions=self.extensions,
            transform=self.transform,
            anomaly_source_labels=self.anomaly_source_labels,
            mask_root=self.mask_root,
            mask_labels=self.mask_labels,
            require_masks_for_mask_labels=self.require_masks_for_mask_labels,
            skip_samples_without_masks_for_mask_labels=self.skip_samples_without_masks_for_mask_labels,
            restrict_normal_to_mask_classes=self.restrict_normal_to_mask_classes,
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
        if self.stratify_object_classes:
            normal_by_class: dict[str, list[int]] = {}
            for idx in normal_indices:
                sample = full_dataset.samples[idx]
                paths = sample.get("paths", {})
                ref_path = paths.get("thermal") or next(iter(paths.values()))
                object_class = self._infer_object_class_from_path(Path(ref_path)) or "__unknown__"
                normal_by_class.setdefault(object_class, []).append(idx)

            train_norm_idx: list[int] = []
            val_norm_idx: list[int] = []
            test_norm_idx: list[int] = []

            for object_class in sorted(normal_by_class.keys()):
                idxs = normal_by_class[object_class]
                rng.shuffle(idxs)
                n = len(idxs)
                n_train_c = int(n * tr)
                n_val_c = int(n * vr)
                train_norm_idx.extend(idxs[:n_train_c])
                val_norm_idx.extend(idxs[n_train_c:n_train_c + n_val_c])
                test_norm_idx.extend(idxs[n_train_c + n_val_c:])

            rng.shuffle(train_norm_idx)
            rng.shuffle(val_norm_idx)
            rng.shuffle(test_norm_idx)
        else:
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
        if self.stratify_object_classes:
            anom_by_class: dict[str, list[int]] = {}
            for idx in anom_indices:
                sample = full_dataset.samples[idx]
                paths = sample.get("paths", {})
                ref_path = paths.get("thermal") or next(iter(paths.values()))
                object_class = self._infer_object_class_from_path(Path(ref_path)) or "__unknown__"
                anom_by_class.setdefault(object_class, []).append(idx)

            val_anom_idx: list[int] = []
            test_anom_idx: list[int] = []
            for object_class in sorted(anom_by_class.keys()):
                idxs = anom_by_class[object_class]
                rng.shuffle(idxs)
                n = len(idxs)
                n_val_c = int(n * va)
                val_anom_idx.extend(idxs[:n_val_c])
                test_anom_idx.extend(idxs[n_val_c:])

            rng.shuffle(val_anom_idx)
            rng.shuffle(test_anom_idx)
        else:
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
        print(
            f"  anomalies ({','.join(self.anomaly_source_labels)}): "
            f"total={M}, val={len(val_anom_idx)}, test={len(test_anom_idx)}"
        )
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

        if self.print_split_per_class:
            train_norm_counts = self._count_object_classes(full_dataset, train_norm_idx)
            val_norm_counts = self._count_object_classes(full_dataset, val_norm_idx)
            test_norm_counts = self._count_object_classes(full_dataset, test_norm_idx)
            val_anom_counts = self._count_object_classes(full_dataset, val_anom_idx)
            test_anom_counts = self._count_object_classes(full_dataset, test_anom_idx)

            print("[PER-CLASS SPLIT]")
            print("  normals:")
            print(f"    train: {self._format_counts(train_norm_counts)}")
            print(f"    val  : {self._format_counts(val_norm_counts)}")
            print(f"    test : {self._format_counts(test_norm_counts)}")
            print(f"  anomalies ({','.join(self.anomaly_source_labels)}):")
            print(f"    val  : {self._format_counts(val_anom_counts)}")
            print(f"    test : {self._format_counts(test_anom_counts)}")

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