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

    Unterstützte Verzeichnisstrukturen:
        root/
            normal/
                thermal/**
                rgb/**
            anomalous/
                thermal/**
                rgb/**

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

    def _resolve_modality_dir(self, label_name: str, modality: str) -> Path:
        """Resolve the directory of a modality for a given label.

        Supports both ``root/label/modality`` and ``root/modality/label``.
        """
        candidates = (
            self.root / label_name / modality,
            self.root / modality / label_name,
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Expected modality folder in one of: "
            f"{candidates[0]} or {candidates[1]}"
        )

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

        for label_name, label in label_map.items():
            modality_files: dict[str, dict[str, Path]] = {}
            pairing_key_sets: list[set[str]] = []

            for modality in self.modalities:
                mod_dir = self._resolve_modality_dir(label_name, modality)

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
                self.samples.append(
                    {
                        "paths": paths_for_modalities,
                        "label": label,
                        "filename": pairing_key,
                        "pairing_key": pairing_key,
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