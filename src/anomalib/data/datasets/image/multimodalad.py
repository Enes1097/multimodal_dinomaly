from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence, Any, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform


class MultiModalFolderDataset(Dataset):
    """Multimodales Dataset für unsplittete Struktur.

    Erwartete Verzeichnisstruktur:
        root/
            thermal/
                normal/**
                anomalous/**
            rgb/
                normal/**
                anomalous/**

    Dateien werden über einen Pairing-Key zusammengeführt. Der Pairing-Key
    wird standardmäßig aus den letzten beiden numerischen Bestandteilen des
    Dateinamens abgeleitet. Damit werden z. B. die folgenden Dateien korrekt
    zusammengeführt:

    - thermal: inspection_payload_thermal_camera_image_<sec>_<nsec>.png
    - rgb: rgb_image_<rgb_sec>_<rgb_nsec>_<sec>_<nsec>.png

    Für Modalitäten mit nur einem Timestamp im Namen werden ebenfalls die
    letzten beiden numerischen Bestandteile verwendet.

    Args:
        root: Wurzelverzeichnis des Datensatzes.
        modalities: Modalitäts-Namen = Unterordner unter normal/anomalous.
        extensions: erlaubte Dateiendungen.
        mask_dir: Optionales Verzeichnis mit Ground-Truth-Masken für thermische
            Bilder. Masken werden über denselben Pairing-Key wie die Thermalbilder
            zugeordnet. Fehlende Masken sind erlaubt.
        transform: Optionaler Transform, der auf jede Modalität angewendet wird.
    """
    def __init__(
        self,
        root: str | Path,
        modalities: Sequence[str] = ("thermal", "rgb"),
        extensions: Sequence[str] = (".png", ".jpg", ".jpeg"),
        mask_dir: str | Path | None = None,
        transform: Transform | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.modalities = list(modalities)
        self.extensions = tuple(e.lower() for e in extensions)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.transform = transform

        self.samples: list[Dict[str, Any]] = []
        self._build_index()

    def _extract_pairing_key(self, path: Path) -> str:
        """Extract the modality-independent pairing key from a filename.

        The last timestamp in the filename is used for pairing. For ROS-style
        timestamps this corresponds to the last two numeric groups
        ``<sec>_<nsec>``. If fewer numeric groups are present, a best-effort
        fallback is used.
        """
        numeric_parts = re.findall(r"\d+", path.stem)
        if len(numeric_parts) >= 2:
            return "_".join(numeric_parts[-2:])
        if len(numeric_parts) == 1:
            return numeric_parts[0]
        return path.stem

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

        mask_files: dict[str, Path] = {}
        if self.mask_dir is not None:
            if not self.mask_dir.exists():
                raise FileNotFoundError(f"Expected mask folder: {self.mask_dir}")

            for path in sorted(self.mask_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in self.extensions:
                    continue

                pairing_key = self._extract_pairing_key(path)
                existing_path = mask_files.get(pairing_key)
                if existing_path is not None:
                    chosen_path = min(
                        (existing_path, path),
                        key=lambda candidate: (len(candidate.name), candidate.name),
                    )
                    mask_files[pairing_key] = chosen_path
                    print(
                        f"[WARN] Duplicate mask pairing key '{pairing_key}'. "
                        f"Using '{chosen_path.name}'."
                    )
                    continue

                mask_files[pairing_key] = path

        for label_name, label in label_map.items():
            modality_files: dict[str, dict[str, Path]] = {}
            pairing_key_sets: list[set[str]] = []

            for modality in self.modalities:
                mod_dir = self.root / modality / label_name
                if not mod_dir.exists():
                    raise FileNotFoundError(f"Expected modality folder: {mod_dir}")

                files_for_mod: dict[str, Path] = {}
                for path in sorted(mod_dir.rglob("*")):
                    if not path.is_file():
                        continue
                    if path.suffix.lower() not in self.extensions:
                        continue

                    pairing_key = self._extract_pairing_key(path)
                    existing_path = files_for_mod.get(pairing_key)
                    if existing_path is not None:
                        chosen_path = min(
                            (existing_path, path),
                            key=lambda candidate: (len(candidate.name), candidate.name),
                        )
                        files_for_mod[pairing_key] = chosen_path
                        print(
                            f"[WARN] Duplicate pairing key '{pairing_key}' "
                            f"for label='{label_name}', modality='{modality}'. "
                            f"Using '{chosen_path.name}'."
                        )
                        continue

                    files_for_mod[pairing_key] = path

                modality_files[modality] = files_for_mod
                pairing_key_sets.append(set(files_for_mod.keys()))

            if not pairing_key_sets:
                continue

            common_pairing_keys = set.intersection(*pairing_key_sets)
            if len(common_pairing_keys) == 0:
                print(f"[WARN] No common pairing keys for label '{label_name}'")

            for pairing_key in sorted(common_pairing_keys):
                paths_for_modalities = {
                    modality: modality_files[modality][pairing_key]
                    for modality in self.modalities
                }
                mask_path = mask_files.get(pairing_key) if label_name == "anomalous" else None
                self.samples.append(
                    {
                        "paths": paths_for_modalities,
                        "label": label,
                        "filename": pairing_key,
                        "pairing_key": pairing_key,
                        "label_name": label_name,
                        "mask_path": mask_path,
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

    def _load_mask(self, path: Path) -> torch.Tensor:
        """Load a binary mask tensor from disk.

        The returned tensor has shape (H, W) and values in {0, 1} whenever the
        source mask is stored as a conventional binary image.
        """
        mask = Image.open(path).convert("L")
        mask_tensor = torch.as_tensor(list(mask.getdata()), dtype=torch.uint8).reshape(mask.height, mask.width)
        return (mask_tensor > 0).to(torch.uint8)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_info = self.samples[index]
        paths: dict[str, Path] = sample_info["paths"]
        label: int = sample_info["label"]
        mask_path: Path | None = sample_info.get("mask_path")

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

        for modality, tensor in images.items():
            out[modality] = tensor

        if "thermal" in images:
            out["image"] = images["thermal"]
        elif len(images) == 1:
            out["image"] = next(iter(images.values()))
            
        out["label"] = torch.tensor(label, dtype=torch.long)
        reference_path = paths["thermal"] if "thermal" in paths else next(iter(paths.values()))
        out["image_path"] = str(reference_path)

        if mask_path is not None:
            out["mask"] = self._load_mask(mask_path)
            out["mask_path"] = str(mask_path)
        else:
            out["mask"] = None
            out["mask_path"] = None

        return out