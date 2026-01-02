"""
Download MTG card images from Scryfall for training.

This script:
1. Fetches bulk data from Scryfall API
2. Downloads card images (normal size)
3. Organizes by card name for training
"""

import json
import os
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import hashlib


# Scryfall API endpoints
BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
RATE_LIMIT_DELAY = 0.1  # 100ms between requests (Scryfall requirement)

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
METADATA_FILE = DATA_DIR / "cards_metadata.json"


def get_bulk_data_url():
    """Get the URL for the default cards bulk data."""
    print("Fetching bulk data metadata...")
    response = requests.get(BULK_DATA_URL)
    response.raise_for_status()

    bulk_data = response.json()
    for item in bulk_data["data"]:
        if item["type"] == "default_cards":
            print(f"Found default_cards: {item['download_uri']}")
            print(f"  Updated: {item['updated_at']}")
            print(f"  Size: {item['size'] / 1024 / 1024:.1f} MB")
            return item["download_uri"]

    raise ValueError("Could not find default_cards bulk data")


def download_bulk_data(url):
    """Download and parse the bulk card data."""
    cache_file = DATA_DIR / "bulk_cards.json"

    if cache_file.exists():
        print(f"Using cached bulk data: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    print("Downloading bulk card data (this may take a few minutes)...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Save to cache
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(cache_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = 100 * downloaded / total_size
                print(f"\r  Downloaded: {downloaded / 1024 / 1024:.1f} MB ({pct:.1f}%)", end="")

    print("\n  Done!")

    with open(cache_file, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_cards(cards):
    """
    Filter cards to get unique, usable entries for training.

    We want:
    - Cards with images
    - English language
    - Normal cards (not tokens, emblems, etc.)
    - Group by oracle_name (canonical card name)
    """
    print(f"Filtering {len(cards)} cards...")

    # Group by oracle_name (handles different printings)
    by_name = defaultdict(list)

    skipped = defaultdict(int)

    for card in cards:
        # Skip non-English
        if card.get("lang") != "en":
            skipped["non-english"] += 1
            continue

        # Skip tokens, emblems, etc.
        layout = card.get("layout", "")
        if layout in ["token", "emblem", "art_series", "double_faced_token"]:
            skipped["token/emblem"] += 1
            continue

        # Skip cards without images
        if "image_uris" not in card:
            # Check for card_faces (double-faced cards)
            if "card_faces" in card and card["card_faces"]:
                if "image_uris" not in card["card_faces"][0]:
                    skipped["no-image"] += 1
                    continue
            else:
                skipped["no-image"] += 1
                continue

        # Use oracle_name if available, otherwise name
        card_name = card.get("oracle_name") or card.get("name", "Unknown")

        # Get image URL (prefer 'normal' size, ~488x680)
        if "image_uris" in card:
            image_url = card["image_uris"].get("normal") or card["image_uris"].get("large")
        else:
            # Double-faced card - use front face
            image_url = card["card_faces"][0]["image_uris"].get("normal")

        if not image_url:
            skipped["no-normal-image"] += 1
            continue

        by_name[card_name].append({
            "id": card["id"],
            "name": card_name,
            "set": card.get("set", ""),
            "collector_number": card.get("collector_number", ""),
            "image_url": image_url,
            "layout": layout,
        })

    print(f"  Skipped: {dict(skipped)}")
    print(f"  Unique card names: {len(by_name)}")
    print(f"  Total printings: {sum(len(v) for v in by_name.values())}")

    return by_name


def download_image(card_info, images_dir):
    """Download a single card image."""
    # Create filename from card name (sanitized)
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in card_info["name"])
    safe_name = safe_name[:50]  # Limit length

    # Use card ID to ensure uniqueness
    filename = f"{safe_name}_{card_info['id'][:8]}.jpg"
    filepath = images_dir / filename

    if filepath.exists():
        return filepath, True  # Already downloaded

    try:
        response = requests.get(card_info["image_url"], timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        return filepath, False  # Newly downloaded

    except Exception as e:
        return None, str(e)


def download_images(cards_by_name, max_per_card=5, max_total=None):
    """
    Download card images.

    Args:
        cards_by_name: Dict of card_name -> list of printings
        max_per_card: Max images per unique card name
        max_total: Max total images to download (None for all)
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Build download queue
    download_queue = []
    for card_name, printings in cards_by_name.items():
        # Take up to max_per_card printings per card
        for printing in printings[:max_per_card]:
            download_queue.append(printing)
            if max_total and len(download_queue) >= max_total:
                break
        if max_total and len(download_queue) >= max_total:
            break

    print(f"\nDownloading {len(download_queue)} images...")

    downloaded = 0
    skipped = 0
    errors = 0

    # Download with thread pool (respect rate limit)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}

        for i, card_info in enumerate(download_queue):
            future = executor.submit(download_image, card_info, IMAGES_DIR)
            futures[future] = card_info

            # Rate limiting
            if i % 10 == 0:
                time.sleep(RATE_LIMIT_DELAY * 10)

        for future in as_completed(futures):
            card_info = futures[future]
            result, status = future.result()

            if result is None:
                errors += 1
            elif status is True:
                skipped += 1
            else:
                downloaded += 1

            total = downloaded + skipped + errors
            if total % 100 == 0:
                print(f"  Progress: {total}/{len(download_queue)} "
                      f"(new: {downloaded}, cached: {skipped}, errors: {errors})")

    print(f"\nDownload complete!")
    print(f"  New downloads: {downloaded}")
    print(f"  Already cached: {skipped}")
    print(f"  Errors: {errors}")

    return downloaded + skipped


def create_metadata(cards_by_name):
    """Create metadata file mapping card names to image files."""
    metadata = {}

    for card_name, printings in cards_by_name.items():
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in card_name)
        safe_name = safe_name[:50]

        # Find downloaded images for this card
        images = []
        for printing in printings:
            filename = f"{safe_name}_{printing['id'][:8]}.jpg"
            filepath = IMAGES_DIR / filename
            if filepath.exists():
                images.append({
                    "filename": filename,
                    "set": printing["set"],
                    "id": printing["id"],
                })

        if images:
            metadata[card_name] = {
                "images": images,
                "count": len(images),
            }

    # Save metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {METADATA_FILE}")
    print(f"  Total unique cards: {len(metadata)}")
    print(f"  Total images: {sum(m['count'] for m in metadata.values())}")

    return metadata


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download MTG card images from Scryfall")
    parser.add_argument("--max-images", type=int, default=1000,
                        help="Maximum total images to download (default: 1000)")
    parser.add_argument("--max-per-card", type=int, default=3,
                        help="Maximum images per unique card name (default: 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("MTG Card Image Downloader")
    print("=" * 60)
    print()

    # Step 1: Get bulk data URL
    bulk_url = get_bulk_data_url()

    # Step 2: Download bulk data
    cards = download_bulk_data(bulk_url)

    # Step 3: Filter and group cards
    cards_by_name = filter_cards(cards)

    # Step 4: Download images
    print(f"\nDownloading up to {args.max_images} images (max {args.max_per_card} per card)...")
    download_images(cards_by_name, max_per_card=args.max_per_card, max_total=args.max_images)

    # Step 5: Create metadata
    create_metadata(cards_by_name)

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  1. Run training: python train.py --epochs 30 --batch-size 32")
    print("=" * 60)


if __name__ == "__main__":
    main()
