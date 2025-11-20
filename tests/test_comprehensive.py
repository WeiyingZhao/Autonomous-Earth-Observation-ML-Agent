"""
Comprehensive Test Suite for ML Reproduction Agent.

Tests:
1. Unit tests for core utilities
2. Integration tests for end-to-end critical paths
3. Negative tests for robustness
4. Performance/resource tests

Run with:
    pytest tests/test_comprehensive.py -v
    pytest tests/test_comprehensive.py -v -k "integration"  # Only integration tests
    pytest tests/test_comprehensive.py -v --cov=src        # With coverage
"""

import pytest
import os
import sys
import json
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.state import (
    AgentState,
    PaperSpec,
    DatasetInfo,
    CodeArtifacts,
    TrainingResults,
    ReportInfo,
    PhaseStatus,
    init_phase_tracking,
    update_phase_status,
    should_retry_phase
)
from src.agent.router import LLMRouter
from src.tools.dataset_resolver import DatasetResolver
from src.tools.code_synthesizer import CodeSynthesizer
from src.tools.mock_llm import MockLLM, create_mock_router
from src.data.synthetic_dataset import SyntheticEODataset, create_dataloaders


# ============================================================================
# UNIT TESTS: Core Utilities
# ============================================================================

class TestStateManagement:
    """Unit tests for state management."""

    def test_agent_state_initialization(self):
        """Test AgentState can be created with valid inputs."""
        state = AgentState(
            paper_uri="/path/to/paper.pdf",
            task_hint="classification",
            max_gpu_hours=4.0
        )

        assert state.paper_uri == "/path/to/paper.pdf"
        assert state.task_hint == "classification"
        assert state.max_gpu_hours == 4.0
        assert len(state.errors) == 0
        assert state.paper_spec is None  # Not yet populated

    def test_phase_tracking_initialization(self):
        """Test phase tracking is properly initialized."""
        state = AgentState(paper_uri="test.pdf")
        state = init_phase_tracking(state)

        expected_phases = [
            "parse_paper", "validate_spec", "find_code",
            "resolve_dataset", "build_env", "synthesize_code",
            "prepare_data", "sanity_check", "full_train",
            "evaluate", "generate_report"
        ]

        for phase_name in expected_phases:
            assert phase_name in state.phases
            assert state.phases[phase_name].status == PhaseStatus.PENDING

    def test_phase_status_update(self):
        """Test phase status can be updated."""
        state = AgentState(paper_uri="test.pdf")
        state = init_phase_tracking(state)

        # Update to in_progress
        state = update_phase_status(state, "parse_paper", PhaseStatus.IN_PROGRESS)
        assert state.phases["parse_paper"].status == PhaseStatus.IN_PROGRESS
        assert state.phases["parse_paper"].start_time is not None

        # Update to completed
        state = update_phase_status(state, "parse_paper", PhaseStatus.COMPLETED)
        assert state.phases["parse_paper"].status == PhaseStatus.COMPLETED
        assert state.phases["parse_paper"].end_time is not None

    def test_retry_logic(self):
        """Test retry count and retry decision."""
        state = AgentState(paper_uri="test.pdf")
        state = init_phase_tracking(state)

        phase_name = "parse_paper"

        # Initially should retry
        assert should_retry_phase(state, phase_name) == True

        # Increment retry count
        state = update_phase_status(state, phase_name, PhaseStatus.RETRYING)
        assert state.phases[phase_name].retry_count == 1

        # Max retries is 3, so should still retry
        state = update_phase_status(state, phase_name, PhaseStatus.RETRYING)
        state = update_phase_status(state, phase_name, PhaseStatus.RETRYING)
        assert state.phases[phase_name].retry_count == 3
        assert should_retry_phase(state, phase_name) == False  # Max retries reached


class TestPaperSpec:
    """Unit tests for PaperSpec model."""

    def test_paper_spec_creation(self):
        """Test PaperSpec can be created with minimal fields."""
        spec = PaperSpec(
            title="Test Paper",
            tasks=["classification"],
            sensors=["Sentinel-2"]
        )

        assert spec.title == "Test Paper"
        assert "classification" in spec.tasks
        assert "Sentinel-2" in spec.sensors
        assert spec.abstract is None  # Optional field

    def test_paper_spec_defaults(self):
        """Test PaperSpec default values."""
        spec = PaperSpec(title="Test")

        assert spec.tasks == []
        assert spec.sensors == []
        assert spec.data_requirements == {}
        assert spec.method == {}
        assert spec.metrics == []


class TestDatasetInfo:
    """Unit tests for DatasetInfo model."""

    def test_dataset_info_creation(self):
        """Test DatasetInfo can be created."""
        dataset = DatasetInfo(
            name="EuroSAT",
            source="torchgeo",
            task_type="classification",
            resolution_m=10.0,
            num_classes=10
        )

        assert dataset.name == "EuroSAT"
        assert dataset.source == "torchgeo"
        assert dataset.num_classes == 10

    def test_dataset_info_defaults(self):
        """Test DatasetInfo default values."""
        dataset = DatasetInfo(name="TestDataset", source="custom")

        assert dataset.task_type == "classification"  # Default
        assert dataset.modality == ["optical"]  # Default
        assert dataset.license == "Unknown"  # Default


class TestDatasetResolver:
    """Unit tests for dataset resolution."""

    def test_catalog_loading(self):
        """Test dataset catalog is loaded correctly."""
        resolver = DatasetResolver()

        assert len(resolver.dataset_catalog) > 0
        assert isinstance(resolver.dataset_catalog, list)

        # Check for known datasets
        names = [ds["name"] for ds in resolver.dataset_catalog]
        assert "BigEarthNet" in names
        assert "EuroSAT" in names

    def test_heuristic_matching_classification(self):
        """Test heuristic matching for classification tasks."""
        resolver = DatasetResolver()

        matches = resolver.find_matching_datasets(
            task_type="classification",
            modality=["optical"],
            resolution_m=10,
            sensor="Sentinel-2",
            top_k=3
        )

        assert len(matches) > 0
        assert matches[0]["match_score"] > 0

        # Should match Sentinel-2 datasets
        first_match = matches[0]
        assert "Sentinel-2" in first_match.get("sensor", "") or "optical" in first_match["modality"]

    def test_heuristic_matching_segmentation(self):
        """Test heuristic matching for segmentation tasks."""
        resolver = DatasetResolver()

        matches = resolver.find_matching_datasets(
            task_type="segmentation",
            modality=["optical"],
            top_k=3
        )

        assert len(matches) > 0

        # At least one result should be a segmentation dataset
        segmentation_count = sum(1 for m in matches if m["task_type"] == "segmentation")
        assert segmentation_count > 0


class TestCodeSynthesizer:
    """Unit tests for code generation."""

    def test_project_structure_creation(self):
        """Test project structure is created correctly."""
        synthesizer = CodeSynthesizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = synthesizer.create_project_structure(tmpdir)

            # Check all expected directories exist
            assert os.path.exists(paths["root"])
            assert os.path.exists(paths["src"])
            assert os.path.exists(paths["scripts"])
            assert os.path.exists(paths["configs"])
            assert os.path.exists(paths["tests"])

            # Check __init__.py files
            assert os.path.exists(os.path.join(paths["src"], "__init__.py"))

    def test_template_code_generation(self):
        """Test template-based code generation produces valid code."""
        synthesizer = CodeSynthesizer()

        code_files = synthesizer._generate_template_code(
            task_type="classification",
            dataset_name="EuroSAT",
            model_architecture="resnet18",
            bands=["B02", "B03", "B04"],
            num_classes=10,
            method={"batch_size": 16, "learning_rate": 0.001}
        )

        # Check required files are generated
        required_files = [
            "src/data/dataset.py",
            "src/models/model.py",
            "scripts/train.py",
            "configs/default.yaml"
        ]

        for filepath in required_files:
            assert filepath in code_files, f"Missing required file: {filepath}"

        # Check code contains expected content
        assert "EODataModule" in code_files["src/data/dataset.py"]
        assert "EOModel" in code_files["src/models/model.py"]
        assert "batch_size: 16" in code_files["configs/default.yaml"]

    def test_code_syntax_validity(self):
        """Test generated code has valid Python syntax."""
        synthesizer = CodeSynthesizer()

        code_files = synthesizer._generate_template_code(
            task_type="classification",
            dataset_name="EuroSAT",
            model_architecture="resnet18",
            bands=["B02", "B03", "B04"],
            num_classes=10,
            method={}
        )

        # Try to compile Python files
        for filepath, code_content in code_files.items():
            if filepath.endswith(".py"):
                try:
                    compile(code_content, filepath, 'exec')
                except SyntaxError as e:
                    pytest.fail(f"Syntax error in {filepath}: {e}")


class TestMockLLM:
    """Unit tests for mock LLM."""

    def test_mock_llm_initialization(self):
        """Test MockLLM can be created."""
        llm = MockLLM(task_type="general")
        assert llm is not None
        assert llm.task_type == "general"

    def test_mock_llm_paper_parsing(self):
        """Test MockLLM returns valid paper parsing response."""
        llm = MockLLM(task_type="paper_parsing")

        # Generate response
        response = llm._get_response("Parse this paper", "paper_parsing")

        # Should be valid JSON
        parsed = json.loads(response)
        assert "title" in parsed
        assert "tasks" in parsed
        assert "sensors" in parsed

    def test_mock_router(self):
        """Test mock router provides mock LLMs."""
        router = create_mock_router()

        llm = router.get_model("paper_parsing")
        assert isinstance(llm, MockLLM)

        info = router.get_provider_info("code_generation")
        assert info["provider"] == "mock"


# ============================================================================
# INTEGRATION TESTS: End-to-End Flows
# ============================================================================

class TestEndToEndIntegration:
    """Integration tests for critical paths."""

    def test_synthetic_dataset_loading(self):
        """Test synthetic dataset can be loaded and used."""
        # Generate tiny dataset
        from scripts.generate_synthetic_data import generate_synthetic_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "dataset")

            # Generate small dataset
            generate_synthetic_dataset(
                output_dir=dataset_dir,
                num_samples=10,
                height=32,
                width=32,
                num_bands=4,
                num_classes=3,
                seed=42
            )

            # Load dataset
            dataset = SyntheticEODataset(data_dir=dataset_dir, split="train")

            assert len(dataset) > 0

            # Get a sample
            image, label = dataset[0]

            assert image.shape[0] == 4  # 4 bands
            assert image.shape[1] == 32  # height
            assert image.shape[2] == 32  # width
            assert label.shape == (32, 32)

    def test_dataloader_creation(self):
        """Test dataloaders can be created from synthetic dataset."""
        from scripts.generate_synthetic_data import generate_synthetic_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "dataset")

            generate_synthetic_dataset(
                output_dir=dataset_dir,
                num_samples=10,
                height=32,
                width=32,
                seed=42
            )

            # Create dataloaders
            dataloaders = create_dataloaders(
                data_dir=dataset_dir,
                batch_size=4,
                num_workers=0
            )

            assert "train" in dataloaders
            assert dataloaders["train"] is not None

            # Get a batch
            images, labels = next(iter(dataloaders["train"]))
            assert images.shape[0] <= 4  # Batch size
            assert images.shape[1] == 4  # Bands

    @pytest.mark.skipif(
        not os.path.exists("./test_data/papers"),
        reason="Requires test data from generate_synthetic_data.py"
    )
    def test_agent_with_mock_paper(self):
        """Test agent can process a mock paper specification."""
        from src.agent.nodes import validate_spec_node, resolve_dataset_node

        # Load a mock paper
        paper_path = "./test_data/papers/paper_01_classification.json"
        if not os.path.exists(paper_path):
            pytest.skip("Test data not generated")

        with open(paper_path, 'r') as f:
            paper_spec_dict = json.load(f)

        # Create state
        state = AgentState(
            paper_uri=paper_path,
            task_hint="classification"
        )
        state = init_phase_tracking(state)
        state.paper_spec = PaperSpec(**paper_spec_dict)

        # Run validation
        state = validate_spec_node(state)
        assert state.paper_spec is not None

        # Run dataset resolution
        state = resolve_dataset_node(state)
        assert state.dataset_info is not None


# ============================================================================
# NEGATIVE TESTS: Robustness & Error Handling
# ============================================================================

class TestNegativeScenarios:
    """Tests for error handling and edge cases."""

    def test_invalid_paper_uri(self):
        """Test handling of invalid paper URI."""
        state = AgentState(
            paper_uri="/nonexistent/paper.pdf",
            task_hint="classification"
        )

        # Should not crash when creating state
        assert state.paper_uri == "/nonexistent/paper.pdf"

    def test_missing_paper_spec(self):
        """Test handling when paper spec is missing."""
        from src.agent.nodes import validate_spec_node

        state = AgentState(paper_uri="test.pdf")
        state = init_phase_tracking(state)
        # paper_spec is None

        # Should handle gracefully
        state = validate_spec_node(state)
        assert "No paper spec available" in str(state.errors) or state.phases["validate_spec"].status == PhaseStatus.FAILED

    def test_no_matching_dataset(self):
        """Test dataset resolution when no matches found."""
        resolver = DatasetResolver()

        matches = resolver.find_matching_datasets(
            task_type="unsupported_task",
            modality=["quantum"],  # Non-existent modality
            resolution_m=0.00001,  # Unrealistic resolution
            top_k=3
        )

        # Should return something (even if low quality match)
        assert isinstance(matches, list)

    def test_malformed_paper_spec(self):
        """Test handling of malformed paper specification."""
        # Missing required 'title' field
        with pytest.raises((ValueError, TypeError)):
            spec = PaperSpec()  # title is required

    def test_empty_dataset_catalog(self):
        """Test resolver behavior with empty catalog."""
        resolver = DatasetResolver()

        # Temporarily empty catalog
        original_catalog = resolver.dataset_catalog
        resolver.dataset_catalog = []

        matches = resolver.find_matching_datasets(
            task_type="classification",
            top_k=3
        )

        assert len(matches) == 0

        # Restore catalog
        resolver.dataset_catalog = original_catalog

    def test_code_generation_with_missing_params(self):
        """Test code generation with missing parameters."""
        synthesizer = CodeSynthesizer()

        # Should handle gracefully with defaults
        code_files = synthesizer._generate_template_code(
            task_type="classification",
            dataset_name="Unknown",
            model_architecture="resnet18",
            bands=[],  # Empty bands
            num_classes=0,  # Invalid num_classes
            method={}
        )

        # Should still generate files (even if not perfect)
        assert len(code_files) > 0


# ============================================================================
# PERFORMANCE & RESOURCE TESTS
# ============================================================================

class TestPerformance:
    """Tests for performance and resource usage."""

    def test_dataset_generation_speed(self):
        """Test dataset generation completes in reasonable time."""
        from scripts.generate_synthetic_data import generate_synthetic_dataset
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            start = time.time()

            generate_synthetic_dataset(
                output_dir=tmpdir,
                num_samples=20,
                height=32,
                width=32,
                seed=42
            )

            elapsed = time.time() - start

            # Should complete in < 5 seconds for 20 small samples
            assert elapsed < 5.0, f"Dataset generation too slow: {elapsed:.2f}s"

    def test_dataset_loading_speed(self):
        """Test dataset loading is fast."""
        from scripts.generate_synthetic_data import generate_synthetic_dataset
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "dataset")

            generate_synthetic_dataset(
                output_dir=dataset_dir,
                num_samples=10,
                height=32,
                width=32,
                seed=42
            )

            # Time dataset loading
            start = time.time()
            dataset = SyntheticEODataset(data_dir=dataset_dir, split="train")
            _ = dataset[0]
            elapsed = time.time() - start

            # Should be very fast (< 0.1s)
            assert elapsed < 0.1

    def test_memory_usage_synthetic_dataset(self):
        """Test synthetic dataset has reasonable memory footprint."""
        from scripts.generate_synthetic_data import generate_synthetic_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "dataset")

            # Generate dataset
            generate_synthetic_dataset(
                output_dir=dataset_dir,
                num_samples=50,
                height=64,
                width=64,
                num_bands=4,
                seed=42
            )

            # Check total size
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(dataset_dir)
                for filename in filenames
            )

            # Should be < 10MB for 50 small samples
            assert total_size < 10 * 1024 * 1024, f"Dataset too large: {total_size / 1024 / 1024:.2f} MB"


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_paper_spec():
    """Sample paper specification for testing."""
    return PaperSpec(
        title="Deep Learning for Land Cover Classification",
        tasks=["classification"],
        sensors=["Sentinel-2"],
        data_requirements={
            "bands": ["B02", "B03", "B04", "B08"],
            "gsd_m": 10,
            "patch_size": 64
        },
        method={
            "model_family": "ResNet",
            "backbone": "resnet18",
            "batch_size": 16,
            "learning_rate": 0.001,
            "epochs": 20
        },
        metrics=["accuracy", "f1"],
        baselines=["AlexNet", "VGG16"],
        datasets_mentioned=["EuroSAT"]
    )


@pytest.fixture
def sample_dataset_info():
    """Sample dataset info for testing."""
    return DatasetInfo(
        name="EuroSAT",
        source="torchgeo",
        dataset_id="torchgeo.datasets.EuroSAT",
        task_type="classification",
        modality=["optical"],
        resolution_m=10,
        license="CC-BY-4.0",
        num_classes=10,
        bands=["B02", "B03", "B04", "B08"],
        loader_type="torchgeo"
    )


@pytest.fixture
def mock_agent_state(sample_paper_spec, sample_dataset_info):
    """Complete agent state for testing."""
    state = AgentState(
        paper_uri="/test/paper.pdf",
        task_hint="classification",
        max_gpu_hours=2.0
    )
    state = init_phase_tracking(state)
    state.paper_spec = sample_paper_spec
    state.dataset_info = sample_dataset_info

    return state


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
