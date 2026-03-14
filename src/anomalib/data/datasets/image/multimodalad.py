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
    letzten beiden numerischen Bestandteile verwendet. Beispiel: ambient_temp_images

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

    def _extract_pairing_key(self, path: Path) -> str:
        """Extract the modality-independent pairing key from a filename.

        The last timestamp in the filename is used for pairing. For ROS-style
        timestamps this corresponds to the last two numeric groups
        ``<sec>_<nsec>``. If fewer numeric groups are present, a best-effort
        fallback is used.
        """
        numeric_parts = re.findall(r"\d+", path.stem) #look for numerical parts
        if len(numeric_parts) >= 2:
            return "_".join(numeric_parts[-2:]) #if more or equal to two numerical parts present then use the last two
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

        for label_name, label in label_map.items(): #run once for normal and once for anomalous
            modality_files: dict[str, dict[str, Path]] = {}
            pairing_key_sets: list[set[str]] = [] #saves pairing keys as a list of sets, one set for each modality

            for modality in self.modalities: # go over every modality
                mod_dir = self.root / modality / label_name # datasets/curated/fridge/thermal/normal
                if not mod_dir.exists():
                    raise FileNotFoundError(f"Expected modality folder: {mod_dir}")

                files_for_mod: dict[str, Path] = {} #saves the pairing key and the path of the image
                for path in sorted(mod_dir.rglob("*")): #search in path for files, sort them and iterate over all of them
                    if not path.is_file(): #Ignore directories
                        continue
                    if path.suffix.lower() not in self.extensions: #Only files with png, jpg etc.
                        continue

                    pairing_key = self._extract_pairing_key(path) #extract thermal timestamp from filename
                    existing_path = files_for_mod.get(pairing_key)
                    if existing_path is not None: #this should not happen as two images in the same timestamp are not present. It prevents loading duplicate images if I accidentaly sorted both images twice
                        chosen_path = min(
                            (existing_path, path),
                            key=lambda candidate: (len(candidate.name), candidate.name),
                        ) #Chooses the path with the shorter filename (is absolutely arbitrary)
                        files_for_mod[pairing_key] = chosen_path
                        print(
                            f"[WARN] Duplicate pairing key '{pairing_key}' "
                            f"for label='{label_name}', modality='{modality}'. "
                            f"Using '{chosen_path.name}'."
                        )
                        continue

                    files_for_mod[pairing_key] = path

                modality_files[modality] = files_for_mod #save every image of one modality under its pairing key (thermal timestamp)
                pairing_key_sets.append(set(files_for_mod.keys()))

            if not pairing_key_sets: #if for example we don't have any anomalous images, it can be skipped
                continue

            common_pairing_keys = set.intersection(*pairing_key_sets) #Checks for intersection between timestamps of the different sets and keeps only matching timestamps
            if len(common_pairing_keys) == 0:
                print(f"[WARN] No common pairing keys for label '{label_name}'")

            for pairing_key in sorted(common_pairing_keys):
                paths_for_modalities = { #create dict where for a matching timestamp the image path for each modality is saved 
                    modality: modality_files[modality][pairing_key]
                    for modality in self.modalities
                }
                self.samples.append( #add matching modality paths to dict and save as list entry in self.samples
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
        '''
        unique_labels = sorted({sample["label"] for sample in self.samples})
        print(f"[SANITY] unique labels: {unique_labels}")
        print("[SANITY] first 10 samples:")
        for sample in self.samples[:10]:
            path_names = {modality: path.name for modality, path in sample["paths"].items()}
            print(
                f"  label_name={sample['label_name']}, label={sample['label']}, "
                f"pairing_key={sample['pairing_key']}, "
                f"paths={path_names}"
            )

        if unique_labels not in ([0], [0, 1], [1]):
            raise ValueError(f"Unexpected labels found: {unique_labels}")
        '''
    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path):
        img = Image.open(path)
        return img.convert("RGB") #converts images to rgb if they are grayscale

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_info = self.samples[index]
        paths: dict[str, Path] = sample_info["paths"] #path for a matching timestamp of each modality
        label: int = sample_info["label"]

        images = {
            modality: self._load_image(p) #Loads all images for each modality for a matching timestamp and saves them in a dict
            for modality, p in paths.items()
        }

        if self.transform is not None:
            images = {
                modality: self.transform(img) #use transform like to.Tensor()
                for modality, img in images.items()
            }

        out: Dict[str, Any] = {} # Initialize dict as output batch

        for modality, tensor in images.items():
            out[modality] = tensor #Add tensors for each modality

        if "thermal" in images:
            out["image"] = images["thermal"] #As dinomaly batches expect "image" key, we save the main modality "thermal" as "image"
        elif len(images) == 1:
            out["image"] = next(iter(images.values())) #If only thermal image exists, use it as "image"
            
        out["label"] = torch.tensor(label, dtype=torch.long) #Set label in batch
        reference_path = paths["thermal"] if "thermal" in paths else next(iter(paths.values())) #Similarly to previous logic use "thermal" image for reference path used in visualization and validation
        out["image_path"] = str(reference_path) #Set reference image path in batch
        # No mask as I don't have ground truth masks

        return out