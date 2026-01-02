"""
Generate embeddings for all card images and build FAISS index.
"""

import json
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset import MTGCardDataset
from model import CardEmbeddingModel


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = CardEmbeddingModel(
        num_classes=checkpoint["num_classes"],
        embedding_dim=checkpoint["embedding_dim"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint["embedding_dim"]


@torch.no_grad()
def generate_embeddings(model, dataset, device, batch_size=32):
    """Generate embeddings for all images in dataset."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    embeddings = []
    labels = []

    for images, batch_labels in tqdm(loader, desc="Generating embeddings"):
        images = images.to(device)
        batch_embeddings = model.get_embedding(images)
        embeddings.append(batch_embeddings.cpu().numpy())
        labels.extend(batch_labels.tolist())

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    return embeddings, labels


def build_faiss_index(embeddings, embedding_dim):
    """Build FAISS index for fast similarity search."""
    # Normalize embeddings (for cosine similarity)
    faiss.normalize_L2(embeddings)

    # Create index
    index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine after normalization)
    index.add(embeddings)

    return index


@torch.no_grad()
def generate_reference_embeddings(model, data_dir: Path, metadata_file: Path, device, batch_size=32):
    """Generate embeddings for reference images (one per card)."""
    # Load metadata
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Setup transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # Build list of (card_name, image_path)
    card_names = []
    image_paths = []
    for card_name, info in metadata.items():
        img_path = data_dir / info["filename"]
        if img_path.exists():
            card_names.append(card_name)
            image_paths.append(img_path)

    print(f"Found {len(card_names)} reference images")

    # Generate embeddings in batches
    embeddings = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating reference embeddings"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        for img_path in batch_paths:
            try:
                with open(img_path, 'rb') as f:
                    img_array = np.frombuffer(f.read(), dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                transformed = transform(image=image)
                batch_images.append(transformed["image"])
            except Exception as e:
                # Use zeros for failed images
                batch_images.append(torch.zeros(3, 224, 224))

        # Stack and process batch
        batch_tensor = torch.stack(batch_images).to(device)
        batch_embeddings = model.get_embedding(batch_tensor)
        embeddings.append(batch_embeddings.cpu().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings, card_names


def test_retrieval(model, train_dataset, val_dataset, device):
    """Test retrieval accuracy."""
    print("\nGenerating train embeddings...")
    train_embeddings, train_labels = generate_embeddings(model, train_dataset, device)

    print("\nBuilding FAISS index...")
    index = build_faiss_index(train_embeddings.copy(), model.embedding_dim)

    # Test 1: Self-retrieval on training set (should be ~100% if model learned anything)
    print("\n--- Test 1: Training Set Self-Retrieval ---")
    train_query = train_embeddings.copy()
    faiss.normalize_L2(train_query)

    # Search for nearest neighbors (k=2 to skip self)
    distances, indices = index.search(train_query, 2)

    # Check if second neighbor (first is self) has same label
    correct = 0
    for i, (nn_indices, label) in enumerate(zip(indices, train_labels)):
        # Skip self (index 0), check index 1
        nn_label = train_labels[nn_indices[1]]
        if nn_label == label:
            correct += 1

    train_retrieval_acc = correct / len(train_labels)
    print(f"  Same-card retrieval: {100 * train_retrieval_acc:.2f}%")
    print(f"  (If a card appears multiple times, can we find another instance?)")

    # Test 2: Validation set (unseen cards)
    print("\n--- Test 2: Validation Set (Unseen Cards) ---")
    val_embeddings, val_labels = generate_embeddings(model, val_dataset, device)
    faiss.normalize_L2(val_embeddings)

    distances, indices = index.search(val_embeddings, 5)

    top1_correct = 0
    top5_correct = 0

    for i, (nn_indices, val_label) in enumerate(zip(indices, val_labels)):
        nn_labels = [train_labels[idx] for idx in nn_indices]
        if nn_labels[0] == val_label:
            top1_correct += 1
        if val_label in nn_labels:
            top5_correct += 1

    top1_acc = top1_correct / len(val_labels)
    top5_acc = top5_correct / len(val_labels)

    print(f"  Top-1 Accuracy: {100 * top1_acc:.2f}% (expected ~0% for unseen cards)")
    print(f"  Top-5 Accuracy: {100 * top5_acc:.2f}%")

    return train_retrieval_acc, top1_acc


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).parent / "checkpoints" / "final_model.pt")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).parent / "data" / "images")
    parser.add_argument("--metadata", type=Path,
                        default=Path(__file__).parent / "data" / "cards_metadata.json")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parent / "output")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--reference", action="store_true",
                        help="Build index from reference images (full 32K card database)")
    args = parser.parse_args()

    # Override paths for reference mode
    if args.reference:
        args.data_dir = Path(__file__).parent / "data" / "reference_images"
        args.metadata = Path(__file__).parent / "data" / "reference_metadata.json"
        print("=" * 60)
        print("REFERENCE MODE: Building full card database index")
        print("=" * 60)

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

    if args.reference:
        # Reference mode: build full card database index
        print(f"\nGenerating embeddings for reference images...")
        print(f"  Data dir: {args.data_dir}")
        print(f"  Metadata: {args.metadata}")

        all_embeddings, card_names = generate_reference_embeddings(
            model, args.data_dir, args.metadata, device
        )

        # Build and save FAISS index
        print("\nBuilding FAISS index...")
        index = build_faiss_index(all_embeddings.copy(), embedding_dim)

        index_path = args.output_dir / "card_embeddings_full.faiss"
        faiss.write_index(index, str(index_path))
        print(f"Saved FAISS index to {index_path}")

        # Save label mapping (card names in order)
        label_mapping = {
            "card_names": card_names,
            "num_cards": len(card_names),
            "embedding_dim": embedding_dim,
        }
        mapping_path = args.output_dir / "label_mapping_full.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(label_mapping, f, indent=2)
        print(f"Saved label mapping to {mapping_path}")

        print("\n" + "=" * 60)
        print("REFERENCE INDEX COMPLETE!")
        print("=" * 60)
        print(f"\nTotal cards indexed: {len(card_names)}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Index size: {index_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"\nFiles saved to {args.output_dir}/")

    else:
        # Original mode: test with training data
        train_dataset = MTGCardDataset(
            data_dir=args.data_dir,
            metadata_file=args.metadata,
            transform=MTGCardDataset.val_transform(),
            split="train",
        )

        val_dataset = MTGCardDataset(
            data_dir=args.data_dir,
            metadata_file=args.metadata,
            transform=MTGCardDataset.val_transform(),
            split="val",
        )

        # Test retrieval accuracy
        train_retrieval_acc, val_top1_acc = test_retrieval(model, train_dataset, val_dataset, device)

        # Generate embeddings for all training images (for deployment)
        print("\n" + "=" * 60)
        print("Generating final embeddings for deployment...")
        print("=" * 60)

        all_embeddings, all_labels = generate_embeddings(model, train_dataset, device)

        # Build and save FAISS index
        index = build_faiss_index(all_embeddings.copy(), embedding_dim)
        faiss.write_index(index, str(args.output_dir / "card_embeddings.faiss"))
        print(f"Saved FAISS index to {args.output_dir / 'card_embeddings.faiss'}")

        # Save label mapping
        label_mapping = {
            "idx_to_name": train_dataset.idx_to_class,
            "labels": all_labels.tolist(),
        }
        with open(args.output_dir / "label_mapping.json", "w") as f:
            json.dump(label_mapping, f, indent=2)
        print(f"Saved label mapping to {args.output_dir / 'label_mapping.json'}")

        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)
        print(f"\nTrain self-retrieval: {100*train_retrieval_acc:.1f}%")
        print(f"(This measures: can we find another image of the same card?)")
        print(f"\nFiles saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
