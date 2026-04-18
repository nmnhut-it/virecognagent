"""
Dataset download and preparation for ViRecognAgent.
Supports: 5CD-AI/Viet-Handwriting-OCR, Cinnamon AI Marathon
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def download_viet_handwriting(output_dir: str = "data/viet_handwriting"):
    """Download 5CD-AI/Viet-Handwriting-OCR from HuggingFace."""
    from datasets import load_dataset

    print("=" * 60)
    print("Downloading 5CD-AI/Viet-Handwriting-OCR from HuggingFace...")
    print("This dataset contains ~23,403 Vietnamese handwriting images.")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    ds = load_dataset("5CD-AI/Viet-Handwriting-OCR")

    # Save images and labels
    for split_name in ds.keys():
        split_dir = os.path.join(output_dir, split_name, "images")
        os.makedirs(split_dir, exist_ok=True)

        labels = []
        split_data = ds[split_name]
        print(f"\nProcessing {split_name} split: {len(split_data)} samples")

        for idx, sample in enumerate(tqdm(split_data, desc=f"Saving {split_name}")):
            img = sample.get("image")
            text = sample.get("text", sample.get("label", ""))

            if img is not None:
                img_path = os.path.join(split_dir, f"{idx:06d}.png")
                img.save(img_path)
                labels.append({
                    "image": f"images/{idx:06d}.png",
                    "text": text,
                    "idx": idx
                })

        # Save labels as JSON
        labels_path = os.path.join(output_dir, split_name, "labels.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump(labels, f, ensure_ascii=False, indent=2)

        print(f"  Saved {len(labels)} samples to {os.path.join(output_dir, split_name)}")

    print(f"\nDataset saved to {output_dir}")
    return output_dir


def download_cinnamon(output_dir: str = "data/cinnamon"):
    """Download Cinnamon AI Marathon Vietnamese handwriting dataset."""
    import gdown

    print("=" * 60)
    print("Downloading Cinnamon AI Marathon dataset...")
    print("This dataset contains 1,838 Vietnamese handwritten address images.")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # The Cinnamon dataset is hosted on Google Drive
    folder_url = "https://drive.google.com/drive/folders/1Qa2YA6w6V5MaNV-qxqZwlkPsJ5lEBajc"

    try:
        gdown.download_folder(folder_url, output=output_dir, quiet=False)
        print(f"\nDataset saved to {output_dir}")
    except Exception as e:
        print(f"\nAutomatic download failed: {e}")
        print(f"Please manually download from: {folder_url}")
        print(f"And extract to: {output_dir}")

    return output_dir


def prepare_evaluation_split(
    data_dir: str,
    output_dir: str = "data/eval_split",
    test_ratio: float = 0.1,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """Create train/val/test splits from a dataset."""
    import random

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load labels
    labels_path = os.path.join(data_dir, "labels.json")
    if not os.path.exists(labels_path):
        # Try to find labels in subdirectories
        for split in ["train", "test", "validation"]:
            p = os.path.join(data_dir, split, "labels.json")
            if os.path.exists(p):
                print(f"Found existing split at {data_dir}/{split}")
                return data_dir
        print(f"No labels.json found in {data_dir}")
        return None

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # Shuffle and split
    random.shuffle(labels)
    n = len(labels)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "test": labels[:n_test],
        "val": labels[n_test:n_test + n_val],
        "train": labels[n_test + n_val:]
    }

    for split_name, split_labels in splits.items():
        split_path = os.path.join(output_dir, f"{split_name}.json")
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_labels, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}: {len(split_labels)} samples")

    return output_dir


def dataset_stats(data_dir: str):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)

    diacritic_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ")

    for split in ["train", "test", "validation", "val"]:
        labels_path = os.path.join(data_dir, split, "labels.json")
        if not os.path.exists(labels_path):
            labels_path = os.path.join(data_dir, f"{split}.json")
        if not os.path.exists(labels_path):
            continue

        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        total_chars = sum(len(l["text"]) for l in labels)
        diacritic_count = sum(
            1 for l in labels for c in l["text"] if c in diacritic_chars
        )
        avg_len = total_chars / len(labels) if labels else 0
        diacritic_ratio = diacritic_count / total_chars if total_chars > 0 else 0

        print(f"\n{split}:")
        print(f"  Samples: {len(labels)}")
        print(f"  Total characters: {total_chars:,}")
        print(f"  Avg text length: {avg_len:.1f} chars")
        print(f"  Diacritic characters: {diacritic_count:,} ({diacritic_ratio:.1%})")
        print(f"  Unique characters: {len(set(''.join(l['text'] for l in labels)))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for ViRecognAgent")
    parser.add_argument(
        "--dataset",
        choices=["viet_handwriting", "cinnamon", "all"],
        default="viet_handwriting",
        help="Which dataset to download"
    )
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    args = parser.parse_args()

    if args.dataset in ("viet_handwriting", "all"):
        d = download_viet_handwriting(os.path.join(args.output_dir, "viet_handwriting"))
        if args.stats:
            dataset_stats(d)

    if args.dataset in ("cinnamon", "all"):
        download_cinnamon(os.path.join(args.output_dir, "cinnamon"))
