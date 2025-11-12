"""
Test suite for the ML Reproduction Agent.
Tests individual components and end-to-end execution.
"""

import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.state import AgentState, PaperSpec, DatasetInfo, init_phase_tracking
from src.agent.router import LLMRouter
from src.tools.paper_ingestor import PaperIngestor
from src.tools.dataset_resolver import DatasetResolver
from src.tools.code_synthesizer import CodeSynthesizer


class TestLLMRouter:
    """Test the multi-provider LLM router."""

    def test_router_initialization(self):
        """Test router can be initialized."""
        router = LLMRouter(default_provider="openai")
        assert router is not None

    def test_provider_selection(self):
        """Test correct provider is selected for each task."""
        router = LLMRouter()

        # Paper parsing should use Claude (long context)
        info = router.get_provider_info("paper_parsing")
        assert info["provider"] == "anthropic"

        # Code generation should use GPT-4
        info = router.get_provider_info("code_generation")
        assert info["provider"] == "openai"

        # Dataset resolution should use Gemini (cost-effective)
        info = router.get_provider_info("dataset_resolution")
        assert info["provider"] == "google"

    def test_cost_estimation(self):
        """Test cost estimation works."""
        router = LLMRouter()
        cost = router.get_cost_estimate("code_generation", 1000, 500)
        assert cost > 0
        assert isinstance(cost, float)


class TestStateManagement:
    """Test state management and tracking."""

    def test_state_initialization(self):
        """Test AgentState can be created."""
        state = AgentState(
            paper_uri="/path/to/paper.pdf",
            task_hint="segmentation"
        )
        assert state.paper_uri == "/path/to/paper.pdf"
        assert state.task_hint == "segmentation"
        assert len(state.errors) == 0

    def test_phase_tracking(self):
        """Test phase tracking initialization."""
        state = AgentState(paper_uri="test.pdf")
        state = init_phase_tracking(state)

        assert "parse_paper" in state.phases
        assert "resolve_dataset" in state.phases
        assert "generate_report" in state.phases


class TestDatasetResolver:
    """Test dataset resolution logic."""

    def test_dataset_catalog(self):
        """Test dataset catalog is loaded."""
        resolver = DatasetResolver()
        assert len(resolver.dataset_catalog) > 0

        # Check for known datasets
        names = [ds["name"] for ds in resolver.dataset_catalog]
        assert "BigEarthNet" in names
        assert "EuroSAT" in names

    def test_heuristic_matching(self):
        """Test heuristic dataset matching."""
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

        # First match should be a good Sentinel-2 classification dataset
        assert "Sentinel-2" in matches[0]["sensor"] or "optical" in matches[0]["modality"]

    def test_resolution_matching(self):
        """Test resolution-based matching."""
        resolver = DatasetResolver()

        # Look for high-resolution aerial datasets
        matches = resolver.find_matching_datasets(
            task_type="segmentation",
            modality=["optical"],
            resolution_m=0.3,
            top_k=3
        )

        assert len(matches) > 0

        # Should match aerial datasets with ~0.3m resolution
        for match in matches:
            if match["resolution_m"]:
                ratio = max(0.3, match["resolution_m"]) / min(0.3, match["resolution_m"])
                # Should be within reasonable tolerance
                if "close_resolution" in match.get("match_reasons", []):
                    assert ratio <= 5


class TestCodeSynthesizer:
    """Test code generation."""

    def test_project_structure_creation(self):
        """Test project structure is created correctly."""
        import tempfile
        import shutil

        synthesizer = CodeSynthesizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = synthesizer.create_project_structure(tmpdir)

            # Check directories exist
            assert os.path.exists(paths["root"])
            assert os.path.exists(paths["src"])
            assert os.path.exists(paths["scripts"])
            assert os.path.exists(paths["configs"])
            assert os.path.exists(paths["tests"])

            # Check __init__.py files
            assert os.path.exists(os.path.join(paths["src"], "__init__.py"))

    def test_template_code_generation(self):
        """Test template-based code generation."""
        synthesizer = CodeSynthesizer()

        code_files = synthesizer._generate_template_code(
            task_type="classification",
            dataset_name="EuroSAT",
            model_architecture="resnet50",
            bands=["B02", "B03", "B04"],
            num_classes=10,
            method={"batch_size": 16, "learning_rate": 0.001}
        )

        # Check required files are generated
        assert "src/data/dataset.py" in code_files
        assert "src/models/model.py" in code_files
        assert "scripts/train.py" in code_files
        assert "configs/default.yaml" in code_files

        # Check code contains expected content
        assert "EODataModule" in code_files["src/data/dataset.py"]
        assert "EOModel" in code_files["src/models/model.py"]
        assert "batch_size: 16" in code_files["configs/default.yaml"]


class TestPaperIngestor:
    """Test paper ingestion."""

    def test_section_splitting(self):
        """Test paper text is split into sections."""
        ingestor = PaperIngestor()

        sample_text = """
Abstract
This is the abstract.

1. Introduction
This is the introduction.

2. Methods
This is the methods section.

3. Experiments
This is the experiments section.

References
[1] Paper citation
"""

        sections = ingestor._split_into_sections(sample_text)

        assert "abstract" in sections
        assert "introduction" in sections
        assert "methods" in sections
        assert "experiments" in sections


# Integration tests
class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY"
    )
    def test_full_pipeline_dry_run(self):
        """Test full pipeline with mock data (no actual LLM calls)."""
        # This would test the complete flow with mocked components
        pass


# Example test data
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
            "patch_size": 256
        },
        method={
            "model_family": "ResNet",
            "backbone": "resnet50",
            "batch_size": 16,
            "learning_rate": 0.001,
            "epochs": 50
        },
        metrics=["accuracy", "f1"],
        baselines=["AlexNet", "VGG"],
        datasets_mentioned=["BigEarthNet"]
    )


@pytest.fixture
def sample_dataset_info():
    """Sample dataset info for testing."""
    return DatasetInfo(
        name="BigEarthNet",
        source="torchgeo",
        dataset_id="torchgeo.datasets.BigEarthNet",
        task_type="classification",
        modality=["optical"],
        resolution_m=10,
        license="CDLA",
        num_classes=43,
        bands=["B02", "B03", "B04", "B08"],
        loader_type="torchgeo"
    )


def test_with_fixtures(sample_paper_spec, sample_dataset_info):
    """Test using fixtures."""
    assert sample_paper_spec.title is not None
    assert sample_dataset_info.name == "BigEarthNet"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
