"""
Generate FAISS index with AUGMENTED reference embeddings.

The Problem:
- Model trained with heavy augmentation (good)
- But FAISS index built from pristine Scryfall scans (bad)
- Webcam captures don't match clean references

The Fix:
- Generate multiple augmented versions of each reference image
- Store all embeddings in index OR compute mean embedding
- This covers the "query space" of real webcam captures
"""

import json
from pathlib import Path
import argparse

import cv2
import faiss
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from model import CardEmbeddingModel


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = CardEmbeddingModel(
        num_classes=checkpoint["num_classes"],
        embedding_dim=checkpoint["embedding_dim"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint["embedding_dim"]


def get_augmentation_pipeline():
    """Heavy augmentation to simulate webcam conditions."""
    return A.Compose([
        # Resize to slightly larger, then crop
        A.Resize(256, 256),
        A.RandomCrop(224, 224),

        # Geometric transforms (simulate angle/perspective)
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=(-10, 10),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.8
        ),
        A.Perspective(scale=(0.02, 0.06), p=0.5),

        # Color transforms (simulate lighting)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.8),

        # Simulate glare (bright spots)
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 1),
            src_radius=50,
            num_flare_circles_lower=1,
            num_flare_circles_upper=2,
            p=0.3
        ),

        # Blur (simulate focus issues)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=5),
        ], p=0.4),

        # Noise (simulate webcam noise)
        A.OneOf([
            A.GaussNoise(std_range=(0.05, 0.15)),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3)),
        ], p=0.4),

        # Quality degradation
        A.ImageCompression(quality_range=(60, 95), p=0.4),

        # Normalize and convert
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_clean_transform():
    """Clean transform (no augmentation) for baseline."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


@torch.no_grad()
def generate_augmented_embeddings(
    model,
    data_dir: Path,
    metadata_file: Path,
    device,
    num_augmentations: int = 10,
    strategy: str = "mean"  # "mean" or "all"
):
    """
    Generate embeddings with augmentation.

    Args:
        strategy: "mean" - average all augmented embeddings per card
                  "all" - keep all augmented embeddings (larger index)
    """
    # Load metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    augment_transform = get_augmentation_pipeline()
    clean_transform = get_clean_transform()

    # Build list of cards
    card_names = []
    image_paths = []
    for card_name, info in metadata.items():
        img_path = data_dir / info["filename"]
        if img_path.exists():
            card_names.append(card_name)
            image_paths.append(img_path)

    print(f"Found {len(card_names)} reference images")
    print(f"Generating {num_augmentations} augmented versions per card")
    print(f"Strategy: {strategy}")

    if strategy == "all":
        print(f"Total embeddings: {len(card_names) * (num_augmentations + 1)}")

    all_embeddings = []
    all_names = []  # Track which embedding belongs to which card

    for i, (card_name, img_path) in enumerate(tqdm(zip(card_names, image_paths), total=len(card_names), desc="Processing cards")):
        try:
            # Load image
            with open(img_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
            continue

        card_embeddings = []

        # Generate clean embedding
        clean_img = clean_transform(image=image)["image"].unsqueeze(0).to(device)
        clean_emb = model.get_embedding(clean_img).cpu().numpy()
        card_embeddings.append(clean_emb[0])

        # Generate augmented embeddings
        for _ in range(num_augmentations):
            aug_img = augment_transform(image=image)["image"].unsqueeze(0).to(device)
            aug_emb = model.get_embedding(aug_img).cpu().numpy()
            card_embeddings.append(aug_emb[0])

        if strategy == "mean":
            # Average all embeddings for this card
            mean_emb = np.mean(card_embeddings, axis=0)
            # Renormalize
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            all_embeddings.append(mean_emb)
            all_names.append(card_name)
        else:  # strategy == "all"
            # Keep all embeddings
            for emb in card_embeddings:
                all_embeddings.append(emb)
                all_names.append(card_name)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    return embeddings, all_names, card_names


def main():
    parser = argparse.ArgumentParser(description="Generate augmented FAISS index")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).parent / "checkpoints" / "best_model.pt")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).parent / "data" / "reference_images")
    parser.add_argument("--metadata", type=Path,
                        default=Path(__file__).parent / "data" / "reference_metadata.json")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "output")
    parser.add_argument("--num-augmentations", type=int, default=10,
                        help="Number of augmented versions per card")
    parser.add_argument("--strategy", type=str, default="mean",
                        choices=["mean", "all"],
                        help="mean: average embeddings, all: keep all embeddings")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, embedding_dim = load_model(args.checkpoint, device)
    print(f"Model loaded (embedding_dim={embedding_dim})")

    # Generate embeddings
    print(f"\nGenerating augmented embeddings...")
    embeddings, embedding_names, unique_card_names = generate_augmented_embeddings(
        model,
        args.data_dir,
        args.metadata,
        device,
        num_augmentations=args.num_augmentations,
        strategy=args.strategy,
    )

    # Build FAISS index
    print(f"\nBuilding FAISS index...")
    print(f"  Embeddings shape: {embeddings.shape}")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)

    # Save index
    suffix = f"_aug{args.num_augmentations}_{args.strategy}"
    index_path = args.output_dir / f"card_embeddings{suffix}.faiss"
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to {index_path}")

    # Save label mapping
    if args.strategy == "mean":
        label_mapping = {
            "card_names": unique_card_names,
            "num_cards": len(unique_card_names),
            "embedding_dim": embedding_dim,
            "augmentations": args.num_augmentations,
            "strategy": args.strategy,
        }
    else:  # "all" strategy - need to track which embedding maps to which card
        label_mapping = {
            "card_names": unique_card_names,  # Unique names for dedup
            "embedding_to_card": embedding_names,  # Which card each embedding belongs to
            "num_cards": len(unique_card_names),
            "num_embeddings": len(embedding_names),
            "embedding_dim": embedding_dim,
            "augmentations": args.num_augmentations,
            "strategy": args.strategy,
        }

    mapping_path = args.output_dir / f"label_mapping{suffix}.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Saved label mapping to {mapping_path}")

    print("\n" + "=" * 60)
    print("AUGMENTED INDEX COMPLETE!")
    print("=" * 60)
    print(f"\nTotal cards: {len(unique_card_names)}")
    print(f"Total embeddings: {embeddings.shape[0]}")
    print(f"Augmentations per card: {args.num_augmentations}")
    print(f"Strategy: {args.strategy}")
    print(f"Index size: {index_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"\nFiles saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
