from __future__ import annotations

from pathlib import Path
from typing import Sequence, Any, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform


class MultiModalFolderDataset(Dataset):
    """Multimodales Dataset für unsplittete Struktur.
    Erwartete Struktur:
        root/
          normal/
            thermal/*.png
            rgb/*.png
          anomalous/
            thermal/*.png
            rgb/*.png
    Args:
        root: Wurzelverzeichnis des Datensatzes.
        modalities: Modalitäts-Namen = Unterordner unter normal/anomalous.
        extensions: erlaubte Dateiendungen.
        transform: Optionaler Transform, der auf jede Modalität angewendet wird.
    """
    def __init__(
        self,
        root: str | Path,
        modalities: Sequence[str] = ("thermal", "rgb"),
        extensions: Sequence[str] = (".png", ".jpg", ".jpeg"),
        transform: Transform | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.modalities = list(modalities)
        self.extensions = tuple(e.lower() for e in extensions)
        self.transform = transform

        self.samples: list[Dict[str, Any]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Baut self.samples mit allen (normal/anomalous)-Samples und Modalitäten.

        Jedes Sample:
            {
                "paths": { modality_name: Path, ... },
                "label": 0 (normal) oder 1 (anomalous),
                "filename": str,
                "label_name": "normal" | "anomalous"
            }
        """
        label_map = {
            "normal": 0,
            "anomalous": 1,
        }

        for label_name, label in label_map.items():
            base_label_dir = self.root / label_name
            if not base_label_dir.exists():
                raise FileNotFoundError(f"Expected folder: {base_label_dir}")

            modality_files: dict[str, dict[str, Path]] = {}
            filename_sets: list[set[str]] = []

            for modality in self.modalities:
                mod_dir = base_label_dir / modality
                if not mod_dir.exists():
                    raise FileNotFoundError(f"Expected modality folder: {mod_dir}")

                files_for_mod: dict[str, Path] = {}
                for path in mod_dir.rglob("*"):
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in self.extensions:
                        continue
                    files_for_mod[path.name] = path

                modality_files[modality] = files_for_mod
                filename_sets.append(set(files_for_mod.keys()))

            if not filename_sets:
                continue

            common_filenames = set.intersection(*filename_sets)
            if len(common_filenames) == 0:
                print(f"[WARN] No common files for label '{label_name}'")

            for fname in sorted(common_filenames):
                paths_for_modalities = {
                    modality: modality_files[modality][fname]
                    for modality in self.modalities
                }
                self.samples.append(
                    {
                        "paths": paths_for_modalities,
                        "label": label,
                        "filename": fname,
                        "label_name": label_name,
                    }
                )

        num_normal = sum(1 for s in self.samples if s["label"] == 0)
        num_anom = sum(1 for s in self.samples if s["label"] == 1)
        print(
            f"[INFO] MultiModalFolderDataset: total={len(self.samples)}, "
            f"normal={num_normal}, anomalous={num_anom}, modalities={self.modalities}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path):
        img = Image.open(path)
        return img.convert("RGB")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_info = self.samples[index]
        paths: dict[str, Path] = sample_info["paths"]
        label: int = sample_info["label"]

        images = {
            modality: self._load_image(p)
            for modality, p in paths.items()
        }

        if self.transform is not None:
            images = {
                modality: self.transform(img)
                for modality, img in images.items()
            }

        out: Dict[str, Any] = {}

        if len(images) == 1:
            out["image"] = next(iter(images.values()))
        else:
            for modality, tensor in images.items():
                out[modality] = tensor
            
        out["label"] = torch.tensor(label, dtype=torch.long)
        out["image_path"] = str(next(iter(paths.values())))
        # No mask as I don't have ground truth masks

        return out