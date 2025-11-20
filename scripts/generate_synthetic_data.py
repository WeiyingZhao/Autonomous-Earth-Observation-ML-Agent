#!/usr/bin/env python3
"""
Synthetic Data Generation Script for ML Reproduction Agent.

Generates:
1. Mock paper specifications (no PDF needed)
2. Tiny synthetic Earth Observation datasets (< 10MB)
3. Reproducible test fixtures

Usage:
    python scripts/generate_synthetic_data.py --output-dir ./test_data
"""

import argparse
import json
import os
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, List


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # Note: torch.manual_seed() would be called if torch is imported


def generate_mock_paper_spec(paper_id: int = 1, task_type: str = "classification") -> Dict[str, Any]:
    """
    Generate a mock paper specification without needing a real PDF.

    Args:
        paper_id: Unique paper identifier
        task_type: Task type (classification, segmentation, detection)

    Returns:
        Dictionary matching PaperSpec schema
    """
    tasks_config = {
        "classification": {
            "title": f"Deep Learning for Land Cover Classification Using Sentinel-2 Imagery (Paper #{paper_id})",
            "tasks": ["classification"],
            "sensors": ["Sentinel-2"],
            "data_requirements": {
                "bands": ["B02", "B03", "B04", "B08"],
                "gsd_m": 10,
                "patch_size": 64,
                "aoi": "Europe"
            },
            "method": {
                "model_family": "CNN",
                "backbone": "resnet18",
                "batch_size": 16,
                "learning_rate": 0.001,
                "epochs": 20,
                "optimizer": "Adam",
                "loss": "CrossEntropyLoss"
            },
            "metrics": ["accuracy", "f1", "precision", "recall"],
            "baselines": ["AlexNet", "VGG16"],
            "datasets_mentioned": ["EuroSAT", "BigEarthNet"]
        },
        "segmentation": {
            "title": f"Semantic Segmentation of Urban Areas Using Multi-Spectral Imagery (Paper #{paper_id})",
            "tasks": ["segmentation"],
            "sensors": ["Sentinel-2", "Landsat-8"],
            "data_requirements": {
                "bands": ["B02", "B03", "B04", "B08", "B11"],
                "gsd_m": 10,
                "patch_size": 256,
                "aoi": "Urban regions"
            },
            "method": {
                "model_family": "U-Net",
                "backbone": "resnet34",
                "batch_size": 8,
                "learning_rate": 0.0001,
                "epochs": 50,
                "optimizer": "AdamW",
                "loss": "DiceLoss"
            },
            "metrics": ["miou", "pixel_accuracy", "dice_score"],
            "baselines": ["FCN", "DeepLabV3"],
            "datasets_mentioned": ["LoveDA", "SpaceNet"]
        },
        "detection": {
            "title": f"Object Detection in Satellite Imagery Using Faster R-CNN (Paper #{paper_id})",
            "tasks": ["detection"],
            "sensors": ["WorldView-3"],
            "data_requirements": {
                "bands": ["R", "G", "B", "NIR"],
                "gsd_m": 0.3,
                "patch_size": 512,
                "aoi": "Global"
            },
            "method": {
                "model_family": "Faster R-CNN",
                "backbone": "resnet50",
                "batch_size": 4,
                "learning_rate": 0.0001,
                "epochs": 100,
                "optimizer": "SGD",
                "loss": "SmoothL1Loss"
            },
            "metrics": ["map", "ap50", "ap75"],
            "baselines": ["YOLO", "SSD"],
            "datasets_mentioned": ["DOTA", "xView"]
        }
    }

    spec = tasks_config.get(task_type, tasks_config["classification"]).copy()

    # Add common fields
    spec.update({
        "abstract": f"This paper presents a novel approach to {task_type} in Earth Observation imagery using deep learning.",
        "equations": ["loss = CE(y_pred, y_true)", "accuracy = (TP + TN) / (TP + TN + FP + FN)"],
        "algorithms": ["Forward pass through CNN", "Backpropagation with Adam optimizer"]
    })

    return spec


def generate_synthetic_eo_image(
    height: int = 64,
    width: int = 64,
    num_bands: int = 4,
    num_classes: int = 5,
    seed: int = 42
) -> tuple:
    """
    Generate a tiny synthetic Earth Observation image and label.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        num_bands: Number of spectral bands (e.g., 4 for RGB + NIR)
        num_classes: Number of land cover classes
        seed: Random seed

    Returns:
        Tuple of (image, label) as numpy arrays
    """
    rng = np.random.RandomState(seed)

    # Generate synthetic multispectral image
    # Simulate realistic band distributions (scaled 0-1)
    image = np.zeros((num_bands, height, width), dtype=np.float32)

    for band in range(num_bands):
        # Add some structure (not pure noise)
        base = rng.rand(height // 4, width // 4)

        # Upsample with bilinear interpolation
        from scipy.ndimage import zoom
        image[band] = zoom(base, 4, order=1)

        # Add some noise
        image[band] += rng.randn(height, width) * 0.1

        # Clip to valid range
        image[band] = np.clip(image[band], 0, 1)

    # Generate synthetic segmentation label
    # Create regions with different classes
    label = np.zeros((height, width), dtype=np.int64)

    # Simple vertical striping pattern
    stripe_width = width // num_classes
    for i in range(num_classes):
        start = i * stripe_width
        end = (i + 1) * stripe_width if i < num_classes - 1 else width
        label[:, start:end] = i

    # Add some noise to make it more realistic
    noise_mask = rng.rand(height, width) > 0.9
    label[noise_mask] = rng.randint(0, num_classes, size=noise_mask.sum())

    return image, label


def generate_synthetic_dataset(
    output_dir: str,
    num_samples: int = 50,
    split_ratios: Dict[str, float] = None,
    height: int = 64,
    width: int = 64,
    num_bands: int = 4,
    num_classes: int = 5,
    seed: int = 42
):
    """
    Generate a complete synthetic EO dataset with train/val/test splits.

    Args:
        output_dir: Directory to save dataset
        num_samples: Total number of samples
        split_ratios: Dictionary with train/val/test ratios (must sum to 1.0)
        height: Image height
        width: Image width
        num_bands: Number of spectral bands
        num_classes: Number of classes
        seed: Random seed
    """
    if split_ratios is None:
        split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

    assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    print(f"Generating synthetic EO dataset:")
    print(f"  Output: {output_dir}")
    print(f"  Samples: {num_samples}")
    print(f"  Image size: {height}x{width}x{num_bands}")
    print(f"  Classes: {num_classes}")
    print(f"  Splits: {split_ratios}")

    # Calculate split sizes
    splits = {}
    remaining = num_samples
    for split_name, ratio in split_ratios.items():
        if split_name == list(split_ratios.keys())[-1]:
            # Last split gets remaining samples
            splits[split_name] = remaining
        else:
            splits[split_name] = int(num_samples * ratio)
            remaining -= splits[split_name]

    # Generate data for each split
    sample_idx = 0
    metadata = {"splits": {}, "num_classes": num_classes, "num_bands": num_bands, "class_names": []}

    # Generate class names
    class_names = [
        "Forest", "Grassland", "Urban", "Water", "Agriculture",
        "Barren", "Wetland", "Snow", "Cloud", "Shadow"
    ]
    metadata["class_names"] = class_names[:num_classes]

    for split_name, split_size in splits.items():
        print(f"\nGenerating {split_name} split: {split_size} samples")

        split_dir = output_path / split_name
        split_dir.mkdir(exist_ok=True)

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        split_metadata = []

        for i in range(split_size):
            # Generate sample
            image, label = generate_synthetic_eo_image(
                height=height,
                width=width,
                num_bands=num_bands,
                num_classes=num_classes,
                seed=seed + sample_idx
            )

            # Save as numpy files (can be loaded easily)
            sample_id = f"sample_{sample_idx:04d}"
            np.save(images_dir / f"{sample_id}.npy", image)
            np.save(labels_dir / f"{sample_id}.npy", label)

            # Record metadata
            split_metadata.append({
                "id": sample_id,
                "image_path": str(images_dir / f"{sample_id}.npy"),
                "label_path": str(labels_dir / f"{sample_id}.npy"),
                "shape": [height, width],
                "num_bands": num_bands
            })

            sample_idx += 1

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{split_size} samples")

        metadata["splits"][split_name] = {
            "size": split_size,
            "samples": split_metadata
        }

    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Dataset generated successfully!")
    print(f"  Metadata: {metadata_path}")

    # Calculate approximate size
    import os
    total_size = sum(
        os.path.getsize(os.path.join(dirpath, filename))
        for dirpath, dirnames, filenames in os.walk(output_dir)
        for filename in filenames
    )
    print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")

    return metadata


def generate_mock_paper_specs(output_dir: str, num_papers: int = 5):
    """
    Generate multiple mock paper specifications.

    Args:
        output_dir: Directory to save paper specs
        num_papers: Number of papers to generate
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    task_types = ["classification", "segmentation", "detection"]

    print(f"Generating {num_papers} mock paper specifications:")

    papers = []
    for i in range(num_papers):
        task_type = task_types[i % len(task_types)]
        spec = generate_mock_paper_spec(paper_id=i + 1, task_type=task_type)

        # Save individual paper
        paper_path = output_path / f"paper_{i + 1:02d}_{task_type}.json"
        with open(paper_path, 'w') as f:
            json.dump(spec, f, indent=2)

        papers.append({
            "id": i + 1,
            "task_type": task_type,
            "path": str(paper_path),
            "title": spec["title"]
        })

        print(f"  ✓ {paper_path.name}")

    # Save index
    index_path = output_path / "papers_index.json"
    with open(index_path, 'w') as f:
        json.dump({"papers": papers, "count": len(papers)}, f, indent=2)

    print(f"\n✓ Generated {num_papers} paper specifications")
    print(f"  Index: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for ML Reproduction Agent testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./test_data",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of dataset samples to generate (default: 50)"
    )
    parser.add_argument(
        "--num-papers",
        type=int,
        default=5,
        help="Number of mock paper specs to generate (default: 5)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Size of synthetic images (default: 64x64)"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=5,
        help="Number of land cover classes (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SYNTHETIC DATA GENERATION FOR ML REPRODUCTION AGENT")
    print("=" * 70)
    print()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate mock paper specifications
    print("\n[1/2] Generating mock paper specifications...")
    papers_dir = output_path / "papers"
    generate_mock_paper_specs(str(papers_dir), num_papers=args.num_papers)

    # 2. Generate synthetic EO dataset
    print("\n[2/2] Generating synthetic EO dataset...")
    dataset_dir = output_path / "synthetic_eo_dataset"
    generate_synthetic_dataset(
        output_dir=str(dataset_dir),
        num_samples=args.num_samples,
        height=args.image_size,
        width=args.image_size,
        num_classes=args.num_classes,
        seed=args.seed
    )

    print("\n" + "=" * 70)
    print("✓ ALL DATA GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. View paper specs: cat {papers_dir}/papers_index.json")
    print(f"  2. View dataset metadata: cat {dataset_dir}/metadata.json")
    print(f"  3. Run tests: pytest tests/ -v")
    print(f"  4. Run evaluation: python scripts/evaluate_agent.py --data-dir {args.output_dir}")


if __name__ == "__main__":
    main()
