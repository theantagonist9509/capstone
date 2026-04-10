import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TransformDataset(Dataset):
    """
    Wraps any dataset and applies a transform to its image component.

    Supports both image-only datasets (``__getitem__`` returns a bare image)
    and labelled datasets (``__getitem__`` returns ``(image, label)``).
    The return type mirrors whatever the underlying dataset returns.

    Use together with ``ISIC2018Dataset(transform=None)`` and
    ``torch.utils.data.random_split`` to give each split its own
    augmentation pipeline without duplicating file-loading code::

        base = ISIC2018Dataset(root_dir=..., transform=None, labels_csv=..., include_labels=[...])
        train_sub, val_sub = random_split(base, [n_train, n_val])
        train_ds = TransformDataset(train_sub, train_transform)
        val_ds   = TransformDataset(val_sub,   val_transform)

    Parameters
    ----------
    dataset : Dataset
        Any dataset (or Subset) whose ``__getitem__`` returns either a bare
        image or a ``(image, label)`` tuple.
    transform : callable
        Transform applied to the image before it is returned.
    """

    def __init__(self, dataset, transform):
        self.dataset   = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item["image"]
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": item["label"], "id": item["id"]}


class ISIC2020Dataset(Dataset):
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
        transform=None,
        labels_csv: str = None,
        duplicates_csv: str = None,
    ):
        self.root_dir = root_dir

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

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")  # load from disk on demand
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = img

        stem = os.path.splitext(os.path.basename(img_path))[0]
        label = self.label_map[stem] if self.label_map is not None else None
        return {"image": tensor, "label": label, "id": stem}


class ISIC2018Dataset(Dataset):
    """
    Lazy-loading dataset for the ISIC 2018 Task 3 JPEG images.

    Parameters
    ----------
    root_dir : str
        Directory that contains the image files.
    target_height : int
        Height (pixels) to resize every image to.
    target_width : int
        Width (pixels) to resize every image to.
    extra_transforms : callable, optional
        Additional torchvision transforms applied *after* resize & to-tensor.
    labels_csv : str, optional
        Path to the ground-truth CSV file
        (e.g. ISIC2018_Task3_Validation_GroundTruth.csv).
        Must contain an ``image`` column and one-hot label columns
        (MEL, NV, BCC, AKIEC, BKL, DF, VASC).
    include_labels : list of str, optional
        Label column names to return alongside each image, e.g.
        ``["MEL", "NV", "BCC"]``.  Must be a subset of the seven class
        columns in the CSV.  When ``None`` (the default), no labels are
        returned and ``__getitem__`` yields a bare image tensor.
        Ignored when ``labels_csv`` is ``None``.

    Notes
    -----
    Images are loaded from disk one at a time inside ``__getitem__``, so the
    entire dataset never needs to fit in RAM.
    Only files whose names end with ``.jpg`` or ``.jpeg`` or ``.png`` (case-insensitive)
    are included; every other file is skipped.
    When ``include_labels`` is provided, only images that appear in the CSV
    are included in the dataset.
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    ALL_LABELS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    def __init__(
        self,
        root_dir: str,
        transform=None,
        labels_csv: str = None,
        include_labels: list = [],
    ):
        self.root_dir = root_dir

        # Optionally load labels from the ground-truth CSV
        self.label_map = None
        self.include_labels = []

        if labels_csv is not None and len(include_labels) > 0:
            unknown = set(include_labels) - set(self.ALL_LABELS)
            if unknown:
                raise ValueError(
                    f"Unknown label column(s): {unknown}. "
                    f"Valid columns are: {self.ALL_LABELS}"
                )
            self.include_labels = list(include_labels)
            df = pd.read_csv(labels_csv)
            df = df.set_index("image")
            self.label_map = df[self.include_labels].astype("float32")

        # Collect valid image paths
        all_files = sorted(os.listdir(root_dir))
        valid_stems = set(self.label_map.index) if self.label_map is not None else None

        self.image_paths = [
            os.path.join(root_dir, f)
            for f in all_files
            if os.path.splitext(f)[1].lower() in self.VALID_EXTENSIONS
            and (valid_stems is None or os.path.splitext(f)[0] in valid_stems)
        ]

        if not self.image_paths:
            raise FileNotFoundError(
                f"No {self.VALID_EXTENSIONS} files found in '{root_dir}'. "
                "Check that root_dir is correct."
            )

        print(f"Found {len(self.image_paths)} valid images in '{root_dir}'")

        # Build the transform pipeline
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = img

        stem = os.path.splitext(os.path.basename(img_path))[0]
        if self.label_map is not None:
            label = torch.tensor(
                self.label_map.loc[stem].values, dtype=torch.float32
            )
        else:
            label = None
        return {"image": tensor, "label": label, "id": stem}
