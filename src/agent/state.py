"""
State schema definitions for the ML Agent using Pydantic.
These models define the shared state that flows through the LangGraph.
"""

from typing import Dict, List, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class PhaseStatus(str, Enum):
    """Status of each phase in the pipeline."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class PaperSpec(BaseModel):
    """
    Structured representation of a parsed ML paper.
    Schema matches the spec from readme.md Phase 1.
    """
    title: str
    abstract: Optional[str] = None
    tasks: List[str] = Field(
        default_factory=list,
        description="e.g., 'semantic_segmentation', 'classification', 'detection'"
    )
    sensors: List[str] = Field(
        default_factory=list,
        description="e.g., 'Sentinel-2', 'SAR(S1)', 'Landsat-8'"
    )
    data_requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Bands, GSD, AOI, seasons, etc."
    )
    method: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model families, losses, training regime, hyperparameters"
    )
    metrics: List[str] = Field(
        default_factory=list,
        description="mIoU, F1, OA, AP50, etc."
    )
    baselines: List[str] = Field(
        default_factory=list,
        description="Baseline methods cited in paper"
    )
    datasets_mentioned: List[str] = Field(
        default_factory=list,
        description="Dataset names cited in paper"
    )
    equations: List[str] = Field(
        default_factory=list,
        description="Key equations extracted from paper"
    )
    algorithms: List[str] = Field(
        default_factory=list,
        description="Algorithm pseudocode or descriptions"
    )


class DatasetInfo(BaseModel):
    """
    Information about a resolved dataset.
    Schema matches readme.md Phase 2 output.
    """
    name: str
    source: Literal["huggingface", "radiant_mlhub", "kaggle", "torchgeo", "custom"]
    dataset_id: Optional[str] = None  # e.g., HF dataset ID
    task_type: str = "classification"  # classification, segmentation, detection - with default
    modality: List[str] = Field(default_factory=lambda: ["optical"])  # optical, SAR, multisensor - with default
    resolution_m: Optional[float] = None  # Ground sampling distance
    license: str = "Unknown"  # License info - with default for incomplete LLM responses
    citation: Optional[str] = None
    splits: Dict[str, int] = Field(
        default_factory=dict,
        description="Number of samples per split: {train: X, val: Y, test: Z}"
    )
    bands: List[str] = Field(
        default_factory=list,
        description="Available spectral bands"
    )
    num_classes: Optional[int] = None
    download_size_gb: Optional[float] = None
    loader_type: str = "custom"  # torchgeo, datasets, custom
    metadata_url: Optional[str] = None


class CodeArtifacts(BaseModel):
    """
    Generated code and configuration artifacts.
    Maps to readme.md Phase 3 output.
    """
    project_root: str
    model_architecture: str  # e.g., "deeplabv3plus", "resnet50"
    framework: str = "pytorch-lightning"
    files_generated: List[str] = Field(default_factory=list)
    config_path: Optional[str] = None
    dockerfile_path: Optional[str] = None
    tests_passing: bool = False
    unit_tests_count: int = 0


class TrainingResults(BaseModel):
    """
    Results from training and evaluation.
    Maps to readme.md Phase 5 output.
    """
    checkpoint_path: Optional[str] = None
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Final evaluation metrics: {miou: 0.75, f1: 0.82, ...}"
    )
    training_time_hours: Optional[float] = None
    epochs_completed: int = 0
    best_epoch: Optional[int] = None
    learning_curves: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Training and validation curves"
    )
    confusion_matrix: Optional[List[List[int]]] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None


class ReportInfo(BaseModel):
    """
    Generated reproducibility report.
    Maps to readme.md Phase 6 output.
    """
    report_path: str
    repro_card_path: str
    plots_generated: List[str] = Field(default_factory=list)
    deviations_from_paper: List[str] = Field(
        default_factory=list,
        description="List of differences from original paper"
    )
    citations: List[str] = Field(default_factory=list)


class PhaseTracking(BaseModel):
    """
    Tracks the status and retry count of each phase.
    """
    phase_name: str
    status: PhaseStatus = PhaseStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None


class AgentState(BaseModel):
    """
    Master state for the ML reproduction agent.
    This state is passed between nodes in the LangGraph.
    """
    # Input
    run_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    paper_uri: str  # PDF path or arXiv/DOI link
    task_hint: Optional[str] = None  # Optional user-provided task hint
    max_gpu_hours: float = 6.0
    target_sensors: List[str] = Field(default_factory=list)

    # Parsed paper information
    paper_spec: Optional[PaperSpec] = None

    # Dataset resolution
    dataset_info: Optional[DatasetInfo] = None
    dataset_alternatives: List[DatasetInfo] = Field(
        default_factory=list,
        description="Alternative datasets if primary fails"
    )

    # Code and environment
    code_artifacts: Optional[CodeArtifacts] = None
    repo_found: Optional[str] = None  # Official repo URL if found

    # Training and evaluation
    training_results: Optional[TrainingResults] = None

    # Report
    report_info: Optional[ReportInfo] = None

    # Phase tracking
    phases: Dict[str, PhaseTracking] = Field(
        default_factory=dict,
        description="Status of each phase: parse_paper, resolve_dataset, etc."
    )

    # Artifacts directory
    artifacts_dir: str = Field(
        default_factory=lambda: f"/artifacts/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    # Error tracking
    errors: List[str] = Field(
        default_factory=list,
        description="List of errors encountered during execution"
    )

    # Messages for LLM conversation history (if needed)
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Conversation history for multi-turn interactions"
    )

    # Constraints
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution constraints: max_dataset_size_gb, sandbox_mode, etc."
    )

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


def init_phase_tracking(state: AgentState) -> AgentState:
    """
    Initialize phase tracking for all phases.
    """
    phase_names = [
        "parse_paper",
        "validate_spec",
        "find_code",
        "resolve_dataset",
        "build_env",
        "synthesize_code",
        "prepare_data",
        "sanity_check",
        "full_train",
        "evaluate",
        "generate_report"
    ]

    for phase_name in phase_names:
        if phase_name not in state.phases:
            state.phases[phase_name] = PhaseTracking(phase_name=phase_name)

    return state


def update_phase_status(
    state: AgentState,
    phase_name: str,
    status: PhaseStatus,
    error_message: Optional[str] = None
) -> AgentState:
    """
    Update the status of a specific phase.
    """
    if phase_name in state.phases:
        phase = state.phases[phase_name]
        phase.status = status

        if status == PhaseStatus.IN_PROGRESS:
            phase.start_time = datetime.now()
        elif status in [PhaseStatus.COMPLETED, PhaseStatus.FAILED]:
            phase.end_time = datetime.now()

        if error_message:
            phase.error_message = error_message

        if status == PhaseStatus.RETRYING:
            phase.retry_count += 1

    return state


def should_retry_phase(state: AgentState, phase_name: str) -> bool:
    """
    Determine if a phase should be retried based on retry count.
    """
    if phase_name not in state.phases:
        return False

    phase = state.phases[phase_name]
    return phase.retry_count < phase.max_retries
