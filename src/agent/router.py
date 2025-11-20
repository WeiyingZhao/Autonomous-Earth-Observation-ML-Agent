"""
Multi-provider LLM router for cost and capability optimization.
Routes requests to OpenAI, DeepSeek, or Google Gemini based on task requirements.
"""

from typing import Literal, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
import os


TaskType = Literal[
    "paper_parsing",      # Long context, structure extraction
    "code_generation",    # Complex code synthesis
    "dataset_resolution", # Quick API queries and matching
    "method_planning",    # Decomposition and architecture design
    "report_generation",  # Document generation
    "general"            # Default general-purpose tasks
]


class LLMRouter:
    """
    Intelligent LLM router that selects the best provider for each task type.

    Provider Selection Logic:
    - DeepSeek: Long-context paper parsing (128K tokens, cost-effective)
    - GPT-4: Code generation and complex reasoning
    - Gemini Pro: Quick queries, dataset resolution, reports (cost-effective)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        default_provider: str = "openai"
    ):
        """
        Initialize the LLM router with API keys.

        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            deepseek_api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            google_api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            default_provider: Default provider if task-specific routing fails
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.deepseek_api_key = deepseek_api_key or os.getenv("DEEPSEEK_API_KEY")
        self.google_api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self.default_provider = default_provider

        # Cache initialized models
        self._model_cache: Dict[str, BaseChatModel] = {}

        # Task-to-provider mapping (optimized for cost and capability)
        self.task_routing: Dict[TaskType, str] = {
            "paper_parsing": "deepseek",       # DeepSeek for long context
            "code_generation": "openai",       # GPT-4 for code
            "dataset_resolution": "google",    # Gemini for quick queries
            "method_planning": "openai",       # GPT-4 for planning
            "report_generation": "google",     # Gemini for documents
            "general": self.default_provider
        }

        # Model configurations
        # FIX: Updated to use stable model names and added request timeout
        self.model_configs = {
            "openai": {
                "model": "gpt-4-turbo-preview",
                "temperature": 0.1,
                "max_tokens": 4096,
                "timeout": 120  # 2 minute timeout for LLM calls
            },
            "openai_code": {  # Specialized for code generation
                "model": "gpt-4",
                "temperature": 0,
                "max_tokens": 4096,  # GPT-4 has 8192 TOTAL context (input+output), so max output is ~4096
                "timeout": 180  # 3 minute timeout for code generation
            },
            "deepseek": {
                "model": "deepseek-chat",
                "temperature": 0.1,
                "max_tokens": 4096,
                "base_url": "https://api.deepseek.com",
                "timeout": 120
            },
            "google": {
                "model": "gemini-1.5-pro",  # Using stable version (gemini-2.5-flash not yet widely available)
                "temperature": 0.1,
                "max_tokens": 8192,
                "timeout": 120
            }
        }

    def get_model(
        self,
        task_type: TaskType = "general",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        force_provider: Optional[str] = None
    ) -> BaseChatModel:
        """
        Get the appropriate LLM for a given task type.

        Args:
            task_type: Type of task to perform
            temperature: Override default temperature
            max_tokens: Override default max tokens
            force_provider: Force a specific provider ("openai", "anthropic", "google")

        Returns:
            Configured chat model for the task

        Raises:
            ValueError: If required API key is missing
        """
        # Determine provider
        provider = force_provider or self.task_routing.get(task_type, self.default_provider)

        # Special case: use GPT-4 for code generation
        if task_type == "code_generation" and provider == "openai":
            config_key = "openai_code"
        else:
            config_key = provider

        # Get config
        config = self.model_configs[config_key].copy()

        # Override with custom parameters
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_tokens"] = max_tokens

        # Create cache key
        cache_key = f"{config_key}_{config['temperature']}_{config['max_tokens']}"

        # Return cached model if available
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Initialize new model
        if provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            model = ChatOpenAI(
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                api_key=self.openai_api_key
            )

        elif provider == "deepseek":
            if not self.deepseek_api_key:
                raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")
            model = ChatOpenAI(
                model=config["model"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                api_key=self.deepseek_api_key,
                base_url=config["base_url"]
            )

        elif provider == "google":
            if not self.google_api_key:
                raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
            model = ChatGoogleGenerativeAI(
                model=config["model"],
                temperature=config["temperature"],
                max_output_tokens=config["max_tokens"],
                google_api_key=self.google_api_key
            )

        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Cache and return
        self._model_cache[cache_key] = model
        return model

    def get_cost_estimate(self, task_type: TaskType, input_tokens: int, output_tokens: int) -> float:
        """
        Estimate cost for a given task based on provider pricing (as of 2024).

        Args:
            task_type: Type of task
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        provider = self.task_routing.get(task_type, self.default_provider)

        # Pricing per 1M tokens (approximate, as of 2024)
        pricing = {
            "openai": {"input": 10.0, "output": 30.0},      # GPT-4 Turbo
            "deepseek": {"input": 0.27, "output": 1.10},    # DeepSeek V3
            "google": {"input": 1.25, "output": 5.0}        # Gemini 1.5 Pro
        }

        rates = pricing.get(provider, pricing["openai"])
        cost = (input_tokens / 1_000_000 * rates["input"] +
                output_tokens / 1_000_000 * rates["output"])

        return cost

    def get_provider_info(self, task_type: TaskType) -> Dict[str, Any]:
        """
        Get information about which provider will be used for a task.

        Args:
            task_type: Type of task

        Returns:
            Dict with provider name, model, and capabilities
        """
        provider = self.task_routing.get(task_type, self.default_provider)
        config = self.model_configs.get(provider, self.model_configs[self.default_provider])

        return {
            "provider": provider,
            "model": config["model"],
            "max_tokens": config["max_tokens"],
            "temperature": config["temperature"]
        }


# Singleton instance for easy access
_default_router: Optional[LLMRouter] = None


def get_router() -> LLMRouter:
    """
    Get the default LLM router instance (singleton pattern).

    Returns:
        Default LLMRouter instance
    """
    global _default_router
    if _default_router is None:
        _default_router = LLMRouter()
    return _default_router


def init_router(
    openai_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    default_provider: str = "openai"
) -> LLMRouter:
    """
    Initialize and return the default router with specific API keys.

    Args:
        openai_api_key: OpenAI API key
        deepseek_api_key: DeepSeek API key
        google_api_key: Google API key
        default_provider: Default provider

    Returns:
        Configured LLMRouter instance
    """
    global _default_router
    _default_router = LLMRouter(
        openai_api_key=openai_api_key,
        deepseek_api_key=deepseek_api_key,
        google_api_key=google_api_key,
        default_provider=default_provider
    )
    return _default_router


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    router = LLMRouter()

    # Get model for paper parsing (will use DeepSeek)
    paper_model = router.get_model("paper_parsing")
    print(f"Paper parsing: {router.get_provider_info('paper_parsing')}")

    # Get model for code generation (will use GPT-4)
    code_model = router.get_model("code_generation")
    print(f"Code generation: {router.get_provider_info('code_generation')}")

    # Get model for dataset resolution (will use Gemini)
    dataset_model = router.get_model("dataset_resolution")
    print(f"Dataset resolution: {router.get_provider_info('dataset_resolution')}")

    # Estimate costs
    cost = router.get_cost_estimate("paper_parsing", input_tokens=50000, output_tokens=2000)
    print(f"Estimated cost for paper parsing: ${cost:.4f}")
