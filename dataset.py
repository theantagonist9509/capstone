import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ISICDataset(Dataset):
    """
    Lazy-loading dataset for the ISIC JPEG images.

    Parameters
    ----------
    root_dir : str
        Directory that contains the image files.
    target_height : int
        Height (pixels) to resize every image to.
    target_width : int
        Width  (pixels) to resize every image to.
    extra_transforms : callable, optional
        Additional torchvision transforms applied *after* resize & to-tensor.
    labels_csv : str, optional
        Path to the ground-truth CSV file (e.g. ISIC_2020_Training_GroundTruth.csv).
        Must contain ``image_name`` (filename without extension) and ``target``
        (0 = benign, 1 = malignant) columns.  When provided, ``__getitem__``
        returns a ``(image_tensor, label)`` tuple instead of a bare tensor.
    duplicates_csv : str, optional
        Path to the duplicates CSV file (e.g. ISIC_2020_Training_Duplicates.csv).
        Must contain ``image_name_1`` and ``image_name_2`` columns.  All images
        listed in ``image_name_2`` (the duplicate copies) are excluded from the
        dataset, keeping only one representative from each duplicate pair.

    Notes
    -----
    Images are loaded from disk one at a time inside ``__getitem__``, so the
    entire dataset never needs to fit in RAM.
    Only files whose names end with ``.jpg`` or ``.jpeg`` (case-insensitive)
    are included; every other file is skipped.
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg"}

    def __init__(
        self,
        root_dir: str,
        target_height: int = 128,
        target_width: int = 128,
        extra_transforms=None,
        labels_csv: str = None,
        duplicates_csv: str = None,
    ):
        self.root_dir = root_dir
        self.target_height = target_height
        self.target_width  = target_width

        # Build set of duplicate image names to exclude (image_name_2 entries)
        if duplicates_csv is not None:
            dup_df = pd.read_csv(duplicates_csv, usecols=["image_name_2"])
            excluded_names = set(dup_df["image_name_2"].str.strip())
        else:
            excluded_names = set()

        # Collect valid image paths (skip anything that isn't .jpg/.jpeg or is a duplicate)
        all_files = sorted(os.listdir(root_dir))
        self.image_paths = [
            os.path.join(root_dir, f)
            for f in all_files
            if os.path.splitext(f)[1].lower() in self.VALID_EXTENSIONS
            and os.path.splitext(f)[0] not in excluded_names
        ]

        if not self.image_paths:
            raise FileNotFoundError(
                f"No .jpg/.jpeg files found in '{root_dir}'. "
                "Check that DATASET_DIR is correct."
            )

        print(
            f"Found {len(self.image_paths)} valid JPEG images in '{root_dir}'"
            + (f" ({len(excluded_names)} duplicates excluded)" if excluded_names else "")
        )

        # Optionally load labels from a ground-truth CSV
        if labels_csv is not None:
            df = pd.read_csv(labels_csv, usecols=["image_name", "target"])
            self.label_map = dict(zip(df["image_name"], df["target"].astype(int)))
        else:
            self.label_map = None

        # Build the transform pipeline
        base_tf = [
            transforms.Resize((target_height, target_width)),
            transforms.ToTensor(),                   # → [0, 1] float32
            transforms.Normalize([0.5, 0.5, 0.5],   # → [-1, 1]
                                  [0.5, 0.5, 0.5]),
        ]
        if extra_transforms is not None:
            base_tf.append(extra_transforms)
        self.transform = transforms.Compose(base_tf)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")  # load from disk on demand
        tensor = self.transform(img)
        if self.label_map is not None:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            label = self.label_map[stem]
            return tensor, label
        return tensor