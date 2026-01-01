"""
PyTorch Dataset for MTG card images with augmentation.
"""

import json
import random
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MTGCardDataset(Dataset):
    """Dataset of MTG card images for metric learning."""

    def __init__(
        self,
        data_dir: Path,
        metadata_file: Path,
        transform: Optional[A.Compose] = None,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        """
        Args:
            data_dir: Directory containing card images
            metadata_file: JSON file mapping card names to images
            transform: Albumentations transform pipeline
            split: "train" or "val"
            train_ratio: Fraction of cards for training
            seed: Random seed for reproducible splits
        """
        self.data_dir = Path(data_dir)
        self.transform = transform or self.default_transform()

        # Load metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Create card name to class ID mapping
        all_card_names = sorted(metadata.keys())
        random.seed(seed)
        random.shuffle(all_card_names)

        # Split cards (not images) into train/val
        split_idx = int(len(all_card_names) * train_ratio)
        if split == "train":
            card_names = all_card_names[:split_idx]
        else:
            card_names = all_card_names[split_idx:]

        # Build class mapping and sample list
        self.class_to_idx = {name: idx for idx, name in enumerate(all_card_names)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.num_classes = len(all_card_names)

        # Build list of (image_path, class_idx) samples
        self.samples: List[Tuple[Path, int]] = []
        for card_name in card_names:
            class_idx = self.class_to_idx[card_name]
            for img_info in metadata[card_name]["images"]:
                img_path = self.data_dir / img_info["filename"]
                if img_path.exists():
                    self.samples.append((img_path, class_idx))

        print(f"Dataset [{split}]: {len(self.samples)} images, "
              f"{len(card_names)} cards, {self.num_classes} total classes")

    @staticmethod
    def default_transform():
        """Default augmentation for training."""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    @staticmethod
    def train_transform():
        """Heavy augmentation to simulate webcam conditions."""
        return A.Compose([
            # Resize to slightly larger, then crop
            A.Resize(256, 256),
            A.RandomCrop(224, 224),

            # Geometric transforms (simulate angle/perspective)
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.8, 1.2),
                rotate=(-15, 15),
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.8
            ),
            A.Perspective(scale=(0.02, 0.08), p=0.5),

            # Color transforms (simulate lighting)
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ], p=0.8),

            # Blur (simulate focus issues)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=5),
            ], p=0.3),

            # Noise (simulate webcam noise)
            A.OneOf([
                A.GaussNoise(std_range=(0.05, 0.2)),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3)),
            ], p=0.3),

            # Quality degradation
            A.ImageCompression(quality_range=(70, 100), p=0.3),

            # Normalize and convert
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    @staticmethod
    def val_transform():
        """Minimal augmentation for validation."""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]

        # Load image (handle unicode paths on Windows)
        try:
            # Use numpy to read file, then decode with cv2
            with open(img_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            image = None

        if image is None:
            # Return a random other image if this one fails
            new_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(new_idx)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed["image"]

        return image_tensor, class_idx

    def get_class_name(self, class_idx: int) -> str:
        """Get card name from class index."""
        return self.idx_to_class.get(class_idx, "Unknown")


def create_dataloaders(
    data_dir: Path,
    metadata_file: Path,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Create train and validation dataloaders."""
    train_dataset = MTGCardDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        transform=MTGCardDataset.train_transform(),
        split="train",
    )

    val_dataset = MTGCardDataset(
        data_dir=data_dir,
        metadata_file=metadata_file,
        transform=MTGCardDataset.val_transform(),
        split="val",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.num_classes


if __name__ == "__main__":
    # Test the dataset
    data_dir = Path(__file__).parent / "data" / "images"
    metadata_file = Path(__file__).parent / "data" / "cards_metadata.json"

    if not metadata_file.exists():
        print("Run download_data.py first!")
        exit(1)

    train_loader, val_loader, num_classes = create_dataloaders(
        data_dir, metadata_file, batch_size=8, num_workers=0
    )

    print(f"\nNum classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels[:5]}")
