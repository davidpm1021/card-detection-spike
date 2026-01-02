"""
Download one reference image per unique MTG card for the FAISS index.

This creates the complete card database (~33K unique cards) that will be
used for card identification at inference time.
"""

import json
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Scryfall API endpoints
BULK_DATA_URL = "https://api.scryfall.com/bulk-data"
RATE_LIMIT_DELAY = 0.1  # 100ms between requests (Scryfall requirement)

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
REFERENCE_DIR = DATA_DIR / "reference_images"
REFERENCE_METADATA = DATA_DIR / "reference_metadata.json"


def get_bulk_data_url():
    """Get the URL for the default cards bulk data."""
    print("Fetching bulk data metadata...")
    response = requests.get(BULK_DATA_URL)
    response.raise_for_status()

    bulk_data = response.json()
    for item in bulk_data["data"]:
        if item["type"] == "default_cards":
            print(f"Found default_cards: {item['download_uri']}")
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


def get_unique_cards(cards):
    """
    Get one representative image per unique card name.

    Prioritizes newer printings with high-quality images.
    """
    print(f"Processing {len(cards)} cards...")

    # Group by oracle_name (canonical card name)
    by_name = {}
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

        # Skip digital-only cards
        if card.get("digital", False):
            skipped["digital-only"] += 1
            continue

        # Get image URL
        image_url = None
        if "image_uris" in card:
            image_url = card["image_uris"].get("normal") or card["image_uris"].get("large")
        elif "card_faces" in card and card["card_faces"]:
            if "image_uris" in card["card_faces"][0]:
                image_url = card["card_faces"][0]["image_uris"].get("normal")

        if not image_url:
            skipped["no-image"] += 1
            continue

        # Use oracle_name if available, otherwise name
        card_name = card.get("oracle_name") or card.get("name", "Unknown")

        # Keep only one per card name (prefer newer releases)
        if card_name not in by_name:
            by_name[card_name] = {
                "id": card["id"],
                "name": card_name,
                "set": card.get("set", ""),
                "image_url": image_url,
                "released_at": card.get("released_at", ""),
            }

    print(f"  Skipped: {dict(skipped)}")
    print(f"  Unique cards: {len(by_name)}")

    return by_name


def download_image(card_info, images_dir):
    """Download a single card image."""
    # Create filename from card name (sanitized)
    safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in card_info["name"])
    safe_name = safe_name[:50]

    filename = f"{safe_name}_{card_info['id'][:8]}.jpg"
    filepath = images_dir / filename

    if filepath.exists():
        return filepath, "cached"

    try:
        response = requests.get(card_info["image_url"], timeout=30)
        response.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(response.content)

        return filepath, "downloaded"

    except Exception as e:
        return None, str(e)


def download_reference_images(unique_cards, batch_size=100):
    """Download all reference images with progress tracking."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    cards_list = list(unique_cards.values())
    total = len(cards_list)

    print(f"\nDownloading {total} reference images...")
    print(f"Estimated time: {total * 0.15 / 60:.0f}-{total * 0.25 / 60:.0f} minutes")
    print()

    downloaded = 0
    cached = 0
    errors = 0
    error_cards = []

    start_time = time.time()

    # Process in batches with thread pool
    with ThreadPoolExecutor(max_workers=5) as executor:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = cards_list[batch_start:batch_end]

            futures = {}
            for card_info in batch:
                future = executor.submit(download_image, card_info, REFERENCE_DIR)
                futures[future] = card_info

            for future in as_completed(futures):
                card_info = futures[future]
                result, status = future.result()

                if result is None:
                    errors += 1
                    error_cards.append((card_info["name"], status))
                elif status == "cached":
                    cached += 1
                else:
                    downloaded += 1

            # Progress update
            done = batch_end
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0

            print(f"\r  Progress: {done}/{total} ({100*done/total:.1f}%) | "
                  f"New: {downloaded} | Cached: {cached} | Errors: {errors} | "
                  f"ETA: {eta/60:.1f} min", end="")

            # Rate limiting between batches
            time.sleep(RATE_LIMIT_DELAY * len(batch))

    print()
    print(f"\nDownload complete!")
    print(f"  New downloads: {downloaded}")
    print(f"  Already cached: {cached}")
    print(f"  Errors: {errors}")

    if error_cards:
        print(f"\nFailed cards (first 10):")
        for name, err in error_cards[:10]:
            print(f"  - {name}: {err}")

    return downloaded + cached


def create_reference_metadata(unique_cards):
    """Create metadata file for reference images."""
    metadata = {}

    for card_name, card_info in unique_cards.items():
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in card_name)
        safe_name = safe_name[:50]
        filename = f"{safe_name}_{card_info['id'][:8]}.jpg"
        filepath = REFERENCE_DIR / filename

        if filepath.exists():
            metadata[card_name] = {
                "filename": filename,
                "set": card_info["set"],
                "id": card_info["id"],
            }

    with open(REFERENCE_METADATA, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {REFERENCE_METADATA}")
    print(f"  Total cards with images: {len(metadata)}")

    return metadata


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download reference images for all MTG cards")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of cards (for testing)")
    args = parser.parse_args()

    print("=" * 60)
    print("MTG Reference Image Downloader")
    print("=" * 60)
    print()

    # Step 1: Get bulk data
    bulk_url = get_bulk_data_url()
    cards = download_bulk_data(bulk_url)

    # Step 2: Get unique cards
    unique_cards = get_unique_cards(cards)

    # Apply limit if specified
    if args.limit:
        print(f"\nLimiting to {args.limit} cards (testing mode)")
        unique_cards = dict(list(unique_cards.items())[:args.limit])

    # Step 3: Download images
    download_reference_images(unique_cards)

    # Step 4: Create metadata
    create_reference_metadata(unique_cards)

    print("\n" + "=" * 60)
    print("Done! Next step:")
    print("  python generate_embeddings.py --reference")
    print("=" * 60)


if __name__ == "__main__":
    main()
