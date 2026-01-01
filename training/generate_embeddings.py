"""
Generate embeddings for all card images and build FAISS index.
"""

import json
from pathlib import Path

import faiss
import numpy as np
import torch
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

    # Create datasets
    train_dataset = MTGCardDataset(
        data_dir=args.data_dir,
        metadata_file=args.metadata,
        transform=MTGCardDataset.val_transform(),  # No augmentation for embedding generation
        split="train",
    )

    val_dataset = MTGCardDataset(
        data_dir=args.data_dir,
        metadata_file=args.metadata,
        transform=MTGCardDataset.val_transform(),
        split="val",
    )

    # Test retrieval accuracy
    top1, top5 = test_retrieval(model, train_dataset, val_dataset, device)

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
    print(f"\nTrain self-retrieval: {100*top1:.1f}%")
    print(f"(This measures: can we find another image of the same card?)")
    print(f"\nFiles saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
