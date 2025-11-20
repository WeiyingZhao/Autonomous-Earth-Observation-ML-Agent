"""
Input validation utilities for ML Reproduction Agent.
Provides robust validation and sanitization of user inputs.
"""

import os
import re
from pathlib import Path
from typing import Optional, List
from urllib.parse import urlparse


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_paper_uri(paper_uri: str) -> tuple[str, str]:
    """
    Validate and categorize paper URI.

    Args:
        paper_uri: Paper URI (file path, arXiv ID, or arXiv URL)

    Returns:
        Tuple of (uri_type, normalized_uri)
        - uri_type: "file", "arxiv_id", or "arxiv_url"
        - normalized_uri: Normalized version of the URI

    Raises:
        ValidationError: If URI is invalid
    """
    if not paper_uri or not isinstance(paper_uri, str):
        raise ValidationError("Paper URI cannot be empty")

    paper_uri = paper_uri.strip()

    # Check if it's an arXiv ID (format: arxiv:YYMM.NNNNN or YYMM.NNNNN)
    arxiv_id_pattern = r'^(arxiv:)?(\d{4}\.\d{4,5})(v\d+)?$'
    match = re.match(arxiv_id_pattern, paper_uri, re.IGNORECASE)
    if match:
        arxiv_id = match.group(2)  # Extract the numeric ID
        return "arxiv_id", f"arxiv:{arxiv_id}"

    # Check if it's an arXiv URL
    if "arxiv.org" in paper_uri.lower():
        try:
            parsed = urlparse(paper_uri)
            if parsed.netloc and "arxiv.org" in parsed.netloc.lower():
                # Extract arXiv ID from URL
                path_parts = parsed.path.strip('/').split('/')
                if path_parts:
                    potential_id = path_parts[-1].replace('.pdf', '')
                    # Validate extracted ID
                    if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', potential_id):
                        return "arxiv_url", f"arxiv:{potential_id}"

                raise ValidationError(f"Could not extract valid arXiv ID from URL: {paper_uri}")
        except Exception as e:
            raise ValidationError(f"Invalid arXiv URL: {e}")

    # Otherwise, treat as file path
    path = Path(paper_uri)

    # Check if path is absolute
    if not path.is_absolute():
        # Try to resolve relative path
        path = path.resolve()

    # Validate file extension
    if path.suffix.lower() not in ['.pdf', '.json']:
        raise ValidationError(
            f"Unsupported file type: {path.suffix}. Supported: .pdf, .json"
        )

    # Check if file exists (only for local files)
    if not path.exists():
        raise ValidationError(
            f"File not found: {path}\n"
            f"Please ensure the file exists or provide an arXiv ID (e.g., arxiv:2103.14030)"
        )

    return "file", str(path)


def validate_task_hint(task_hint: Optional[str]) -> Optional[str]:
    """
    Validate task hint.

    Args:
        task_hint: Task type hint

    Returns:
        Normalized task hint or None

    Raises:
        ValidationError: If task hint is invalid
    """
    if task_hint is None:
        return None

    task_hint = task_hint.strip().lower()

    valid_tasks = [
        "classification",
        "segmentation",
        "detection",
        "regression",
        "object_detection",
        "semantic_segmentation",
        "instance_segmentation"
    ]

    # Normalize some common variations
    task_mapping = {
        "classify": "classification",
        "segment": "segmentation",
        "detect": "detection",
        "regress": "regression",
        "object_detection": "detection",
        "semantic_segmentation": "segmentation",
        "instance_segmentation": "segmentation"
    }

    normalized = task_mapping.get(task_hint, task_hint)

    if normalized not in valid_tasks:
        raise ValidationError(
            f"Invalid task hint: '{task_hint}'. "
            f"Supported: {', '.join(valid_tasks)}"
        )

    return normalized


def validate_max_gpu_hours(max_gpu_hours: float) -> float:
    """
    Validate max GPU hours parameter.

    Args:
        max_gpu_hours: Maximum GPU hours

    Returns:
        Validated GPU hours

    Raises:
        ValidationError: If value is invalid
    """
    if not isinstance(max_gpu_hours, (int, float)):
        raise ValidationError("Max GPU hours must be a number")

    if max_gpu_hours <= 0:
        raise ValidationError("Max GPU hours must be positive")

    if max_gpu_hours > 168:  # 1 week
        raise ValidationError(
            f"Max GPU hours too large: {max_gpu_hours}. Maximum: 168 (1 week)"
        )

    return float(max_gpu_hours)


def validate_sensors(sensors: Optional[List[str]]) -> List[str]:
    """
    Validate sensor list.

    Args:
        sensors: List of sensor names

    Returns:
        Validated sensor list

    Raises:
        ValidationError: If sensor list is invalid
    """
    if sensors is None:
        return []

    if not isinstance(sensors, list):
        raise ValidationError("Sensors must be a list")

    # Known sensors (not exhaustive, just for validation)
    known_sensors = [
        "sentinel-1", "sentinel-2", "sentinel-3",
        "landsat-8", "landsat-9",
        "modis", "sar", "optical",
        "worldview-2", "worldview-3",
        "planet", "aerial"
    ]

    validated = []
    for sensor in sensors:
        if not isinstance(sensor, str):
            raise ValidationError(f"Sensor name must be string, got: {type(sensor)}")

        sensor_normalized = sensor.strip().lower()

        if not sensor_normalized:
            continue  # Skip empty strings

        # Warn if sensor is not in known list (but don't fail)
        if sensor_normalized not in known_sensors:
            print(f"⚠️  Warning: Unknown sensor '{sensor}'. Proceeding anyway.")

        validated.append(sensor)

    return validated


def validate_output_dir(output_dir: Optional[str]) -> str:
    """
    Validate and create output directory.

    Args:
        output_dir: Output directory path

    Returns:
        Absolute path to output directory

    Raises:
        ValidationError: If directory cannot be created
    """
    if output_dir is None:
        # Default output dir will be created by agent
        return None

    output_dir = output_dir.strip()

    if not output_dir:
        raise ValidationError("Output directory cannot be empty string")

    # Convert to absolute path
    path = Path(output_dir).resolve()

    # Check if parent directory exists
    if not path.parent.exists():
        raise ValidationError(
            f"Parent directory does not exist: {path.parent}"
        )

    # Try to create directory
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise ValidationError(
            f"Permission denied: Cannot create directory {path}"
        )
    except Exception as e:
        raise ValidationError(
            f"Failed to create output directory: {e}"
        )

    # Check write permissions
    if not os.access(path, os.W_OK):
        raise ValidationError(
            f"No write permission for output directory: {path}"
        )

    return str(path)


def validate_all_inputs(
    paper_uri: str,
    task_hint: Optional[str] = None,
    max_gpu_hours: float = 6.0,
    target_sensors: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> dict:
    """
    Validate all input parameters at once.

    Args:
        paper_uri: Paper URI
        task_hint: Optional task hint
        max_gpu_hours: Maximum GPU hours
        target_sensors: Optional list of sensors
        output_dir: Optional output directory

    Returns:
        Dictionary with validated inputs

    Raises:
        ValidationError: If any validation fails
    """
    validated = {}

    # Validate each parameter
    try:
        uri_type, normalized_uri = validate_paper_uri(paper_uri)
        validated["paper_uri"] = normalized_uri
        validated["paper_uri_type"] = uri_type

        validated["task_hint"] = validate_task_hint(task_hint)
        validated["max_gpu_hours"] = validate_max_gpu_hours(max_gpu_hours)
        validated["target_sensors"] = validate_sensors(target_sensors)
        validated["output_dir"] = validate_output_dir(output_dir)

    except ValidationError as e:
        # Re-raise with context
        raise ValidationError(f"Input validation failed: {e}")

    return validated


# Convenience function for CLI
def validate_cli_args(args) -> dict:
    """
    Validate arguments from argparse.

    Args:
        args: argparse.Namespace object

    Returns:
        Dictionary with validated inputs

    Raises:
        ValidationError: If validation fails
    """
    return validate_all_inputs(
        paper_uri=args.paper,
        task_hint=args.task,
        max_gpu_hours=args.gpu_hours,
        target_sensors=args.sensors,
        output_dir=args.output_dir
    )
