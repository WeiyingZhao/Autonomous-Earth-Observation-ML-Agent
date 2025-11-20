"""
Mock LLM provider for testing without API keys.
Returns realistic responses for different tasks.
"""

import json
from typing import Dict, Any, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class MockLLM(BaseChatModel):
    """
    Mock LLM that returns predefined responses for testing.
    No API keys required.
    """

    task_type: str = Field(default="general", description="Task type for response selection")
    responses: Dict[str, Any] = Field(default_factory=dict, description="Custom responses")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> ChatResult:
        """Generate mock response based on task type."""

        # Get the last user message
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        # Generate response based on task type
        response_text = self._get_response(user_message, self.task_type)

        # Create AI message
        message = AIMessage(content=response_text)

        # Create generation
        generation = ChatGeneration(message=message)

        return ChatResult(generations=[generation])

    def _get_response(self, user_message: str, task_type: str) -> str:
        """Get mock response based on task type and user message."""

        # Check for custom responses first
        if self.responses and user_message in self.responses:
            response = self.responses[user_message]
            if isinstance(response, dict):
                return json.dumps(response, indent=2)
            return str(response)

        # Default responses by task type
        if task_type == "paper_parsing":
            return self._mock_paper_parsing_response(user_message)
        elif task_type == "code_generation":
            return self._mock_code_generation_response(user_message)
        elif task_type == "dataset_resolution":
            return self._mock_dataset_resolution_response(user_message)
        else:
            return "This is a mock LLM response for testing purposes."

    def _mock_paper_parsing_response(self, user_message: str) -> str:
        """Mock response for paper parsing task."""
        paper_spec = {
            "title": "Deep Learning for Land Cover Classification Using Sentinel-2",
            "abstract": "This paper presents a novel CNN-based approach for land cover classification.",
            "tasks": ["classification"],
            "sensors": ["Sentinel-2"],
            "data_requirements": {
                "bands": ["B02", "B03", "B04", "B08"],
                "gsd_m": 10,
                "patch_size": 64
            },
            "method": {
                "model_family": "CNN",
                "backbone": "resnet18",
                "batch_size": 16,
                "learning_rate": 0.001,
                "epochs": 20,
                "optimizer": "Adam"
            },
            "metrics": ["accuracy", "f1", "precision"],
            "baselines": ["AlexNet", "VGG16"],
            "datasets_mentioned": ["EuroSAT"],
            "equations": [],
            "algorithms": []
        }
        return json.dumps(paper_spec, indent=2)

    def _mock_code_generation_response(self, user_message: str) -> str:
        """Mock response for code generation task."""
        code_template = '''
# PyTorch Lightning DataModule
class EODataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Load dataset
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

# PyTorch Lightning Model
class EOModel(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
'''
        return code_template

    def _mock_dataset_resolution_response(self, user_message: str) -> str:
        """Mock response for dataset resolution task."""
        dataset_info = {
            "recommended": "EuroSAT",
            "rationale": "EuroSAT is a good match for Sentinel-2 classification tasks with 10m resolution.",
            "alternatives": ["BigEarthNet", "SEN12MS"]
        }
        return json.dumps(dataset_info, indent=2)

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM."""
        return "mock"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters."""
        return {"task_type": self.task_type}

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Legacy call method (not used in chat models)."""
        return self._get_response(prompt, self.task_type)


def create_mock_router():
    """
    Create a mock router that returns MockLLM instances.
    Use this for testing without API keys.
    """
    class MockRouter:
        def get_model(self, task_type: str = "general"):
            """Get mock LLM for specified task type."""
            return MockLLM(task_type=task_type)

        def get_provider_info(self, task_type: str = "general") -> Dict[str, str]:
            """Get mock provider info."""
            return {
                "provider": "mock",
                "model": "mock-llm",
                "context_window": 8192,
                "cost_per_1k_input": 0.0,
                "cost_per_1k_output": 0.0
            }

        def get_cost_estimate(
            self,
            task_type: str,
            input_tokens: int,
            output_tokens: int
        ) -> float:
            """Mock cost estimate (always $0)."""
            return 0.0

    return MockRouter()


# Singleton instance
_mock_router = None


def get_mock_router():
    """Get or create singleton mock router."""
    global _mock_router
    if _mock_router is None:
        _mock_router = create_mock_router()
    return _mock_router


def init_mock_router():
    """Initialize mock router (for consistency with real router)."""
    return get_mock_router()
