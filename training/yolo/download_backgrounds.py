"""
Background Image Collection for YOLO Training

Downloads and generates diverse background images for synthetic training data.
Sources:
1. DTD (Describable Textures Dataset) - 5,640 texture images
2. Procedural generation - solid colors, gradients, noise patterns

Output: training/data/backgrounds/ with categorized images
"""

import json
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
BACKGROUNDS_DIR = DATA_DIR / "backgrounds"
TEMP_DIR = DATA_DIR / "temp"

DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
MIN_SIZE = 640

# Category mappings from DTD to our categories
DTD_CATEGORY_MAP = {
    # Wood textures
    "banded": "wood",
    "grained": "wood",
    "grooved": "wood",
    "lined": "wood",
    "stratified": "wood",

    # Fabric textures
    "braided": "fabric",
    "woven": "fabric",
    "matted": "fabric",
    "meshed": "fabric",
    "knitted": "fabric",
    "pleated": "fabric",

    # Pattern textures
    "dotted": "pattern",
    "grid": "pattern",
    "honeycombed": "pattern",
    "chequered": "pattern",
    "crosshatched": "pattern",
    "spiralled": "pattern",

    # Desk/surface textures
    "cracked": "desk",
    "weathered": "desk",
    "stained": "desk",
    "scaly": "desk",
    "scratched": "desk",

    # Nature textures
    "leafy": "nature",
    "veined": "nature",
    "marbled": "nature",

    # Misc textures
    "bumpy": "misc",
    "bubbly": "misc",
    "cobwebbed": "misc",
    "crystalline": "misc",
    "flecked": "misc",
    "freckled": "misc",
    "gauzy": "misc",
    "lacelike": "misc",
    "paisley": "misc",
    "perforated": "misc",
    "pitted": "misc",
    "porous": "misc",
    "potholed": "misc",
    "smeared": "misc",
    "sprinkled": "misc",
    "studded": "misc",
    "swirly": "misc",
    "wrinkled": "misc",
    "zigzagged": "misc",
}


class BackgroundCollector:
    """Downloads and generates background images."""

    def __init__(self):
        self.backgrounds_dir = BACKGROUNDS_DIR
        self.temp_dir = TEMP_DIR
        self.stats = {
            "total_count": 0,
            "categories": {},
            "sources": {"dtd": 0, "procedural": 0},
            "stats": {"min_width": 99999, "min_height": 99999, "avg_width": 0, "avg_height": 0}
        }

    def setup_directories(self):
        """Create output directories."""
        print("Setting up directories...")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.backgrounds_dir.mkdir(parents=True, exist_ok=True)

        # Create category subdirectories
        categories = ["wood", "fabric", "solid", "desk", "pattern", "hands", "nature", "misc"]
        for cat in categories:
            (self.backgrounds_dir / cat).mkdir(exist_ok=True)
            self.stats["categories"][cat] = {"count": 0, "files": []}

    def download_with_progress(self, url: str, output_path: Path) -> bool:
        """Download file with progress bar."""
        try:
            print(f"Downloading {url}...")

            # Get file size
            with urllib.request.urlopen(url) as response:
                file_size = int(response.headers.get('Content-Length', 0))

            # Download with progress bar
            with tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                def reporthook(block_num, block_size, total_size):
                    pbar.update(block_size)

                urllib.request.urlretrieve(url, output_path, reporthook=reporthook)

            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False

    def download_dtd(self) -> int:
        """Download and extract DTD dataset."""
        print("\n" + "="*50)
        print("Phase 1: Downloading DTD Dataset")
        print("="*50)

        dtd_archive = self.temp_dir / "dtd.tar.gz"
        dtd_extract = self.temp_dir / "dtd"

        # Download if not exists
        if not dtd_archive.exists():
            if not self.download_with_progress(DTD_URL, dtd_archive):
                print("Failed to download DTD. Skipping...")
                return 0
        else:
            print(f"Using cached DTD archive: {dtd_archive}")

        # Extract
        print("Extracting DTD dataset...")
        dtd_extract.mkdir(exist_ok=True)
        with tarfile.open(dtd_archive, 'r:gz') as tar:
            tar.extractall(self.temp_dir)

        # Find the DTD images directory
        dtd_images_dir = None
        for path in self.temp_dir.rglob("dtd/images"):
            dtd_images_dir = path
            break

        if not dtd_images_dir or not dtd_images_dir.exists():
            print("Could not find DTD images directory. Skipping...")
            return 0

        # Process DTD images
        print("Processing DTD textures...")
        total_processed = 0

        for dtd_category in dtd_images_dir.iterdir():
            if not dtd_category.is_dir():
                continue

            category_name = dtd_category.name
            our_category = DTD_CATEGORY_MAP.get(category_name, "misc")

            images = list(dtd_category.glob("*.jpg"))
            count = 0

            for img_path in images:
                if self.process_and_save_image(img_path, our_category):
                    count += 1
                    total_processed += 1

            if count > 0:
                print(f"  {category_name} -> {our_category}: {count} images")

        self.stats["sources"]["dtd"] = total_processed
        print(f"\nTotal DTD images processed: {total_processed}")
        return total_processed

    def process_and_save_image(self, image_path: Path, category: str) -> bool:
        """Validate, resize if needed, and save image."""
        try:
            img = Image.open(image_path)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Check size
            if img.width < MIN_SIZE or img.height < MIN_SIZE:
                # Skip if way too small
                if img.width < MIN_SIZE // 2 or img.height < MIN_SIZE // 2:
                    return False

                # Resize to minimum size
                scale = MIN_SIZE / min(img.width, img.height)
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Generate output filename
            count = self.stats["categories"][category]["count"]
            output_filename = f"{category}_{count+1:04d}.jpg"
            output_path = self.backgrounds_dir / category / output_filename

            # Save
            img.save(output_path, "JPEG", quality=90)

            # Update stats
            self.stats["categories"][category]["count"] += 1
            self.stats["categories"][category]["files"].append(f"{category}/{output_filename}")
            self.stats["total_count"] += 1

            # Update dimension stats
            self.stats["stats"]["min_width"] = min(self.stats["stats"]["min_width"], img.width)
            self.stats["stats"]["min_height"] = min(self.stats["stats"]["min_height"], img.height)

            return True

        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            return False

    def generate_procedural_backgrounds(self) -> int:
        """Generate procedural backgrounds."""
        print("\n" + "="*50)
        print("Phase 2: Generating Procedural Backgrounds")
        print("="*50)

        total = 0

        # Generate solid colors with noise
        print("Generating solid colors with texture...")
        total += self.generate_solid_colors(100)

        # Generate gradients
        print("Generating gradients...")
        total += self.generate_gradients(50)

        # Generate geometric patterns
        print("Generating geometric patterns...")
        total += self.generate_patterns(50)

        self.stats["sources"]["procedural"] = total
        print(f"\nTotal procedural backgrounds: {total}")
        return total

    def generate_solid_colors(self, count: int) -> int:
        """Generate solid color backgrounds with subtle noise."""
        colors = [
            # Wood tones
            (139, 90, 43), (160, 82, 45), (101, 67, 33), (205, 133, 63),
            # Playmat colors
            (25, 25, 112), (0, 0, 128), (128, 0, 0), (0, 100, 0),
            # Neutral tones
            (128, 128, 128), (169, 169, 169), (192, 192, 192),
            (64, 64, 64), (96, 96, 96), (112, 128, 144),
            # Desk colors
            (245, 245, 220), (255, 250, 240), (211, 211, 211),
        ]

        generated = 0
        for i in range(count):
            color = colors[i % len(colors)]

            # Create base image
            img = np.zeros((640, 640, 3), dtype=np.uint8)
            img[:, :] = color

            # Add random noise
            noise = np.random.randint(-15, 15, (640, 640, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Convert to PIL and add slight blur
            pil_img = Image.fromarray(img)
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=0.5))

            # Save
            count_num = self.stats["categories"]["solid"]["count"]
            output_filename = f"solid_{count_num+1:04d}.jpg"
            output_path = self.backgrounds_dir / "solid" / output_filename
            pil_img.save(output_path, "JPEG", quality=90)

            # Update stats
            self.stats["categories"]["solid"]["count"] += 1
            self.stats["categories"]["solid"]["files"].append(f"solid/{output_filename}")
            self.stats["total_count"] += 1
            generated += 1

        return generated

    def generate_gradients(self, count: int) -> int:
        """Generate gradient backgrounds."""
        generated = 0

        for i in range(count):
            # Random colors
            c1 = tuple(np.random.randint(50, 200, 3).tolist())
            c2 = tuple(np.random.randint(50, 200, 3).tolist())

            # Create gradient
            img = Image.new('RGB', (640, 640))
            draw = ImageDraw.Draw(img)

            # Vertical gradient
            for y in range(640):
                r = int(c1[0] + (c2[0] - c1[0]) * y / 640)
                g = int(c1[1] + (c2[1] - c1[1]) * y / 640)
                b = int(c1[2] + (c2[2] - c1[2]) * y / 640)
                draw.line([(0, y), (640, y)], fill=(r, g, b))

            # Add noise
            img_array = np.array(img)
            noise = np.random.randint(-10, 10, (640, 640, 3), dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            # Determine category based on color
            category = "solid" if i % 2 == 0 else "misc"

            # Save
            count_num = self.stats["categories"][category]["count"]
            output_filename = f"{category}_{count_num+1:04d}.jpg"
            output_path = self.backgrounds_dir / category / output_filename
            img.save(output_path, "JPEG", quality=90)

            # Update stats
            self.stats["categories"][category]["count"] += 1
            self.stats["categories"][category]["files"].append(f"{category}/{output_filename}")
            self.stats["total_count"] += 1
            generated += 1

        return generated

    def generate_patterns(self, count: int) -> int:
        """Generate simple geometric patterns."""
        generated = 0

        for i in range(count):
            img = Image.new('RGB', (640, 640), color=(200, 200, 200))
            draw = ImageDraw.Draw(img)

            pattern_type = i % 4

            if pattern_type == 0:
                # Grid pattern
                spacing = 40
                for x in range(0, 640, spacing):
                    draw.line([(x, 0), (x, 640)], fill=(150, 150, 150), width=1)
                for y in range(0, 640, spacing):
                    draw.line([(0, y), (640, y)], fill=(150, 150, 150), width=1)

            elif pattern_type == 1:
                # Dots pattern
                spacing = 30
                for x in range(spacing // 2, 640, spacing):
                    for y in range(spacing // 2, 640, spacing):
                        draw.ellipse([x-3, y-3, x+3, y+3], fill=(100, 100, 100))

            elif pattern_type == 2:
                # Diagonal lines
                spacing = 20
                for offset in range(-640, 640, spacing):
                    draw.line([(0, offset), (640, 640+offset)], fill=(150, 150, 150), width=2)

            else:
                # Checkerboard
                size = 40
                for x in range(0, 640, size):
                    for y in range(0, 640, size):
                        if (x // size + y // size) % 2 == 0:
                            draw.rectangle([x, y, x+size, y+size], fill=(180, 180, 180))

            # Add noise
            img_array = np.array(img)
            noise = np.random.randint(-10, 10, (640, 640, 3), dtype=np.int16)
            img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            # Save
            count_num = self.stats["categories"]["pattern"]["count"]
            output_filename = f"pattern_{count_num+1:04d}.jpg"
            output_path = self.backgrounds_dir / "pattern" / output_filename
            img.save(output_path, "JPEG", quality=90)

            # Update stats
            self.stats["categories"]["pattern"]["count"] += 1
            self.stats["categories"]["pattern"]["files"].append(f"pattern/{output_filename}")
            self.stats["total_count"] += 1
            generated += 1

        return generated

    def finalize_stats(self):
        """Calculate final statistics."""
        if self.stats["total_count"] > 0:
            # Calculate average dimensions
            total_width = 0
            total_height = 0
            count = 0

            for category, info in self.stats["categories"].items():
                for filepath in info["files"]:
                    full_path = self.backgrounds_dir / filepath
                    if full_path.exists():
                        img = Image.open(full_path)
                        total_width += img.width
                        total_height += img.height
                        count += 1

            if count > 0:
                self.stats["stats"]["avg_width"] = total_width // count
                self.stats["stats"]["avg_height"] = total_height // count

    def save_manifest(self):
        """Save manifest.json."""
        manifest_path = self.backgrounds_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"\nManifest saved to: {manifest_path}")

    def print_summary(self):
        """Print collection summary."""
        print("\n" + "="*50)
        print("Phase 3: Collection Summary")
        print("="*50)

        for category in sorted(self.stats["categories"].keys()):
            count = self.stats["categories"][category]["count"]
            print(f"  {category:12s}: {count:4d} images")

        print("\n" + "="*50)
        print("Collection Complete!")
        print("="*50)
        print(f"Total backgrounds: {self.stats['total_count']}")
        print(f"Sources:")
        print(f"  DTD: {self.stats['sources']['dtd']}")
        print(f"  Procedural: {self.stats['sources']['procedural']}")
        print(f"\nImage stats:")
        print(f"  Min dimensions: {self.stats['stats']['min_width']}x{self.stats['stats']['min_height']}")
        print(f"  Avg dimensions: {self.stats['stats']['avg_width']}x{self.stats['stats']['avg_height']}")

        # Warnings
        if self.stats["categories"]["hands"]["count"] == 0:
            print("\nNote: 'hands' category has 0 images.")
            print("  Consider manual collection if needed for your use case.")

    def run(self):
        """Run the full collection process."""
        print("="*50)
        print("Background Image Collection for YOLO Training")
        print("="*50)

        self.setup_directories()

        # Download DTD
        self.download_dtd()

        # Generate procedural backgrounds
        self.generate_procedural_backgrounds()

        # Finalize
        self.finalize_stats()
        self.save_manifest()
        self.print_summary()


def main():
    collector = BackgroundCollector()
    collector.run()


if __name__ == "__main__":
    main()
