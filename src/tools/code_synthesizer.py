"""
Code Synthesizer Tool - Generates PyTorch Lightning project code.
Creates complete project structure with models, dataloaders, scripts, tests, and configs.
"""

from typing import Dict, Any, Optional, List
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
import yaml
import json
import os
from pathlib import Path


class CodeSynthesizer:
    """
    Synthesizes complete PyTorch Lightning projects from paper and dataset specifications.
    Generates adapters, models, training scripts, configs, and tests.
    """

    def __init__(self, llm=None, prompts_config: Optional[str] = None):
        """
        Initialize Code Synthesizer.

        Args:
            llm: Language model for code generation
            prompts_config: Path to prompts.yml
        """
        self.llm = llm
        self.prompts_config = prompts_config or "src/config/prompts.yml"
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML configuration."""
        if os.path.exists(self.prompts_config):
            with open(self.prompts_config, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def create_project_structure(self, output_dir: str) -> Dict[str, str]:
        """
        Create empty project directory structure.

        Args:
            output_dir: Root directory for the project

        Returns:
            Dict mapping component names to paths
        """
        paths = {
            "root": output_dir,
            "src": os.path.join(output_dir, "src"),
            "src_data": os.path.join(output_dir, "src", "data"),
            "src_models": os.path.join(output_dir, "src", "models"),
            "src_tasks": os.path.join(output_dir, "src", "tasks"),
            "src_utils": os.path.join(output_dir, "src", "utils"),
            "scripts": os.path.join(output_dir, "scripts"),
            "configs": os.path.join(output_dir, "configs"),
            "tests": os.path.join(output_dir, "tests"),
        }

        # Create directories
        for path in paths.values():
            os.makedirs(path, exist_ok=True)

        # Create __init__.py files
        for key in ["src", "src_data", "src_models", "src_tasks", "src_utils"]:
            init_file = os.path.join(paths[key], "__init__.py")
            Path(init_file).touch()

        return paths

    def generate_code_with_llm(
        self,
        paper_spec: Dict[str, Any],
        dataset_info: Dict[str, Any],
        llm=None
    ) -> Dict[str, str]:
        """
        Generate all code components using LLM.

        Args:
            paper_spec: Paper specification
            dataset_info: Dataset information
            llm: Language model to use

        Returns:
            Dict mapping filenames to code content
        """
        if llm is None:
            if self.llm is None:
                from src.agent.router import get_router
                router = get_router()
                llm = router.get_model("code_generation")
            else:
                llm = self.llm

        # Extract key information
        task_type = paper_spec.get("tasks", ["classification"])[0]
        method = paper_spec.get("method", {})
        model_architecture = method.get("model_family", "resnet50")
        backbone = method.get("backbone", "resnet50")

        dataset_name = dataset_info.get("name", "custom_dataset")
        bands = dataset_info.get("bands", ["R", "G", "B"])
        num_classes = dataset_info.get("num_classes", 10)

        # Get code synthesis prompt
        code_prompt = self.prompts.get("code_synthesis", {}).get("prompt", "")

        if not code_prompt:
            code_prompt = """
            Generate PyTorch Lightning code for the task.
            Include DataModule, Model, train script, eval script, config, and tests.
            """

        # Format the prompt with specific parameters
        formatted_prompt = code_prompt.format(
            task_type=task_type,
            dataset_name=dataset_name,
            model_architecture=model_architecture,
            bands=json.dumps(bands),
            patch_size=method.get("patch_size", 256),
            augmentations=json.dumps(method.get("augmentations", ["flip", "rotate"])),
            backbone=backbone,
            losses=json.dumps(method.get("losses", ["cross_entropy"])),
            metrics=json.dumps(paper_spec.get("metrics", ["accuracy"]))
        )

        # Invoke LLM for code generation
        full_prompt = f"""{formatted_prompt}

Return the code for ALL files in the following format:
```python
# FILE: src/data/dataset.py
<code>
```

```python
# FILE: src/models/model.py
<code>
```

... and so on for all required files.
"""

        response = llm.invoke(full_prompt)

        # Parse response to extract files
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)

        code_files = self._parse_code_blocks(response_text)

        # If LLM didn't generate all files, use templates
        if len(code_files) < 5:
            code_files = self._generate_template_code(
                task_type, dataset_name, model_architecture,
                bands, num_classes, method
            )

        return code_files

    def _parse_code_blocks(self, text: str) -> Dict[str, str]:
        """
        Parse code blocks from LLM response.

        Args:
            text: LLM response with code blocks

        Returns:
            Dict mapping file paths to code
        """
        import re

        code_files = {}
        pattern = r'```(?:python)?\s*#\s*FILE:\s*(.+?)\s*\n(.*?)```'

        matches = re.findall(pattern, text, re.DOTALL)

        for filepath, code in matches:
            filepath = filepath.strip()
            code = code.strip()
            code_files[filepath] = code

        return code_files

    def _generate_template_code(
        self,
        task_type: str,
        dataset_name: str,
        model_architecture: str,
        bands: List[str],
        num_classes: int,
        method: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate code from templates (fallback if LLM fails).

        Args:
            task_type: classification, segmentation, detection
            dataset_name: Name of dataset
            model_architecture: Model architecture name
            bands: List of band names
            num_classes: Number of classes
            method: Method specification

        Returns:
            Dict mapping file paths to code
        """
        code_files = {}

        # Dataset module
        code_files["src/data/dataset.py"] = f'''"""
Dataset module for {dataset_name}.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torchgeo


class EODataModule(pl.LightningDataModule):
    """DataModule for {dataset_name}."""

    def __init__(
        self,
        root: str,
        batch_size: int = 16,
        num_workers: int = 4,
        bands: list = {bands},
        patch_size: int = {method.get("patch_size", 256)}
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.bands = bands
        self.patch_size = patch_size

    def prepare_data(self):
        # Download dataset if needed
        pass

    def setup(self, stage=None):
        # Create train/val/test datasets
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
'''

        # Model module
        code_files["src/models/model.py"] = f'''"""
Model definition for {task_type}.
"""

import torch
import torch.nn as nn
from torchvision import models


class EOModel(nn.Module):
    """Model for {task_type}."""

    def __init__(
        self,
        num_classes: int = {num_classes},
        num_bands: int = {len(bands)},
        backbone: str = "{model_architecture}"
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_bands = num_bands

        # Load backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            # Adapt first conv for multispectral
            if num_bands != 3:
                self.backbone.conv1 = nn.Conv2d(
                    num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False
                )

        # Task-specific head
        if "{task_type}" == "classification":
            self.head = nn.Linear(2048, num_classes)
        elif "{task_type}" == "segmentation":
            self.head = nn.Conv2d(2048, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output
'''

        # Training module
        code_files["src/tasks/train_module.py"] = f'''"""
PyTorch Lightning training module.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from src.models.model import EOModel


class EOTrainingModule(pl.LightningModule):
    """Lightning module for training."""

    def __init__(
        self,
        num_classes: int = {num_classes},
        num_bands: int = {len(bands)},
        learning_rate: float = {method.get("learning_rate", 0.001)},
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = EOModel(num_classes, num_bands)
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.train_acc(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.val_acc(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
'''

        # Training script
        code_files["scripts/train.py"] = '''"""
Training script.
"""

import argparse
import yaml
import pytorch_lightning as pl
from src.data.dataset import EODataModule
from src.tasks.train_module import EOTrainingModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Create datamodule
    datamodule = EODataModule(**config["dataset"])

    # Create model
    model = EOTrainingModule(**config["model"])

    # Create trainer
    trainer = pl.Trainer(**config["trainer"])

    # Train
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()
'''

        # Config file
        code_files["configs/default.yaml"] = f'''# Configuration file

dataset:
  root: "data/{dataset_name}"
  batch_size: {method.get("batch_size", 16)}
  num_workers: 4
  bands: {bands}
  patch_size: {method.get("patch_size", 256)}

model:
  num_classes: {num_classes}
  num_bands: {len(bands)}
  learning_rate: {method.get("learning_rate", 0.001)}

trainer:
  max_epochs: {method.get("epochs", 50)}
  accelerator: "gpu"
  devices: 1
  precision: 16
'''

        # Test file
        code_files["tests/test_model.py"] = f'''"""
Tests for model.
"""

import torch
import pytest
from src.models.model import EOModel


def test_model_forward():
    """Test model forward pass."""
    model = EOModel(num_classes={num_classes}, num_bands={len(bands)})
    x = torch.randn(2, {len(bands)}, 256, 256)
    y = model(x)
    assert y.shape[0] == 2
    assert y.shape[1] == {num_classes}


def test_model_overfit_batch():
    """Test model can overfit on single batch."""
    model = EOModel(num_classes={num_classes}, num_bands={len(bands)})
    x = torch.randn(4, {len(bands)}, 64, 64)
    y = torch.randint(0, {num_classes}, (4,))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_loss = None
    for i in range(100):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = torch.nn.functional.cross_entropy(y_pred, y)
        loss.backward()
        optimizer.step()

        if i == 0:
            initial_loss = loss.item()

    final_loss = loss.item()
    assert final_loss < initial_loss * 0.1, "Model should overfit on single batch"
'''

        return code_files

    def synthesize_project(
        self,
        paper_spec: Dict[str, Any],
        dataset_info: Dict[str, Any],
        output_dir: str,
        llm=None
    ) -> Dict[str, Any]:
        """
        Complete code synthesis pipeline.

        Args:
            paper_spec: Paper specification
            dataset_info: Dataset information
            output_dir: Output directory for project
            llm: Language model to use

        Returns:
            Information about generated project
        """
        # Create structure
        paths = self.create_project_structure(output_dir)

        # Generate code
        code_files = self.generate_code_with_llm(paper_spec, dataset_info, llm)

        # Write files
        files_written = []
        for filepath, code in code_files.items():
            full_path = os.path.join(output_dir, filepath)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, 'w') as f:
                f.write(code)

            files_written.append(filepath)

        return {
            "output_dir": output_dir,
            "files_generated": files_written,
            "num_files": len(files_written),
            "project_structure": paths
        }


# LangChain tool wrapper
@tool
def synthesize_code_tool(
    paper_spec: Dict[str, Any],
    dataset_info: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Generate complete PyTorch Lightning project code.

    Args:
        paper_spec: Paper specification with methods and requirements
        dataset_info: Dataset information
        output_dir: Output directory for generated project

    Returns:
        Information about generated project
    """
    synthesizer = CodeSynthesizer()
    result = synthesizer.synthesize_project(paper_spec, dataset_info, output_dir)
    return result


# Example usage
if __name__ == "__main__":
    synthesizer = CodeSynthesizer()

    paper_spec = {
        "tasks": ["classification"],
        "method": {
            "model_family": "resnet50",
            "backbone": "resnet50",
            "batch_size": 16,
            "epochs": 50
        },
        "metrics": ["accuracy", "f1"]
    }

    dataset_info = {
        "name": "EuroSAT",
        "bands": ["B02", "B03", "B04", "B08"],
        "num_classes": 10
    }

    result = synthesizer.synthesize_project(
        paper_spec,
        dataset_info,
        "/tmp/test_project",
        llm=None  # Will use templates
    )

    print(json.dumps(result, indent=2))
