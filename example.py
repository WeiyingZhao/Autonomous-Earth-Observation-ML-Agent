#!/usr/bin/env python3
"""
Example usage of the ML Reproduction Agent.
Demonstrates different ways to use the agent.
"""

import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

from src.agent.graph import create_ml_agent_graph, create_agent_with_tools
from src.agent.state import AgentState, init_phase_tracking
from src.agent.router import get_router, init_router
from src.tools.paper_ingestor import PaperIngestor
from src.tools.dataset_resolver import DatasetResolver
from src.tools.code_synthesizer import CodeSynthesizer


def example_1_basic_usage():
    """
    Example 1: Basic end-to-end reproduction.
    """
    print("=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)

    # Create graph
    graph = create_ml_agent_graph()

    # Initialize state
    state = AgentState(
        paper_uri="arxiv:2103.14030",  # Example arXiv paper
        task_hint="segmentation",
        max_gpu_hours=4.0
    )
    state = init_phase_tracking(state)

    # Run
    print("\n🚀 Running agent...\n")
    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": state.run_id}}
    )

    # Print results
    print("\n📊 Results:")
    print(f"  Paper: {result.paper_spec.title if result.paper_spec else 'N/A'}")
    print(f"  Dataset: {result.dataset_info.name if result.dataset_info else 'N/A'}")
    print(f"  Report: {result.report_info.report_path if result.report_info else 'N/A'}")


def example_2_individual_tools():
    """
    Example 2: Using individual tools separately.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Individual Tools")
    print("=" * 80)

    # Initialize router
    router = init_router()

    # 1. Paper Ingestor
    print("\n1️⃣ Paper Ingestor:")
    ingestor = PaperIngestor(llm=router.get_model("paper_parsing"))

    # Mock paper spec for demonstration
    paper_spec = {
        "title": "Deep Learning for Land Cover Classification",
        "tasks": ["classification"],
        "sensors": ["Sentinel-2"],
        "data_requirements": {
            "bands": ["B02", "B03", "B04", "B08"],
            "gsd_m": 10
        },
        "method": {
            "model_family": "ResNet",
            "backbone": "resnet50"
        },
        "metrics": ["accuracy", "f1"]
    }
    print(f"  ✓ Paper spec created: {paper_spec['title']}")

    # 2. Dataset Resolver
    print("\n2️⃣ Dataset Resolver:")
    resolver = DatasetResolver(llm=router.get_model("dataset_resolution"))
    result = resolver.resolve_dataset(paper_spec, use_llm=False)

    if result.get("recommended"):
        dataset = result["recommended"]
        print(f"  ✓ Dataset found: {dataset['name']}")
        print(f"    Source: {dataset['source']}")
        print(f"    Match score: {dataset.get('match_score', 'N/A')}")
    else:
        print("  ✗ No dataset found")

    # 3. Code Synthesizer
    print("\n3️⃣ Code Synthesizer:")
    synthesizer = CodeSynthesizer(llm=router.get_model("code_generation"))

    # Use template-based generation (no LLM needed)
    code_files = synthesizer._generate_template_code(
        task_type="classification",
        dataset_name=dataset['name'],
        model_architecture="resnet50",
        bands=["B02", "B03", "B04", "B08"],
        num_classes=10,
        method=paper_spec["method"]
    )

    print(f"  ✓ Generated {len(code_files)} files:")
    for filepath in list(code_files.keys())[:5]:
        print(f"    - {filepath}")


def example_3_llm_router():
    """
    Example 3: Using the LLM router.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: LLM Router")
    print("=" * 80)

    router = init_router()

    # Get different models for different tasks
    tasks = ["paper_parsing", "code_generation", "dataset_resolution", "general"]

    for task in tasks:
        info = router.get_provider_info(task)
        cost = router.get_cost_estimate(task, input_tokens=1000, output_tokens=500)

        print(f"\n  Task: {task}")
        print(f"    Provider: {info['provider']}")
        print(f"    Model: {info['model']}")
        print(f"    Estimated cost: ${cost:.4f}")


def example_4_react_agent():
    """
    Example 4: Using ReAct agent (alternative approach).
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: ReAct Agent")
    print("=" * 80)

    # Create ReAct agent with tools
    agent = create_agent_with_tools()

    print("\n  ✓ ReAct agent created with tools:")
    print("    - ingest_paper_tool")
    print("    - resolve_dataset_tool")
    print("    - synthesize_code_tool")

    # Example invocation (would require actual paper)
    # result = agent.invoke({
    #     "messages": [("user", "Reproduce the paper at /path/to/paper.pdf")]
    # })


def example_5_dataset_catalog():
    """
    Example 5: Exploring the dataset catalog.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Dataset Catalog")
    print("=" * 80)

    resolver = DatasetResolver()

    print(f"\n  📊 Total datasets: {len(resolver.dataset_catalog)}")

    # Group by task type
    by_task = {}
    for dataset in resolver.dataset_catalog:
        task = dataset["task_type"]
        by_task[task] = by_task.get(task, 0) + 1

    print("\n  By task type:")
    for task, count in by_task.items():
        print(f"    {task}: {count} datasets")

    # Show some examples
    print("\n  Example datasets:")
    for dataset in resolver.dataset_catalog[:3]:
        print(f"\n    • {dataset['name']}")
        print(f"      Task: {dataset['task_type']}")
        print(f"      Source: {dataset['source']}")
        print(f"      Resolution: {dataset['resolution_m']}m")
        print(f"      Size: {dataset['size_gb']}GB")


def example_6_state_inspection():
    """
    Example 6: Inspecting agent state.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 6: State Inspection")
    print("=" * 80)

    state = AgentState(
        paper_uri="/path/to/paper.pdf",
        task_hint="segmentation",
        max_gpu_hours=6.0,
        target_sensors=["Sentinel-2"]
    )
    state = init_phase_tracking(state)

    print("\n  Initial state:")
    print(f"    Run ID: {state.run_id}")
    print(f"    Paper: {state.paper_uri}")
    print(f"    Task hint: {state.task_hint}")
    print(f"    Max GPU hours: {state.max_gpu_hours}")
    print(f"    Artifacts dir: {state.artifacts_dir}")

    print(f"\n  Phases ({len(state.phases)}):")
    for phase_name, phase in list(state.phases.items())[:5]:
        print(f"    {phase_name}: {phase.status.value}")


def main():
    """
    Run all examples.
    """
    print("\n")
    print("*" * 80)
    print("ML REPRODUCTION AGENT - EXAMPLES")
    print("*" * 80)

    try:
        # Run examples (skip those requiring actual papers/API keys)
        example_2_individual_tools()
        example_3_llm_router()
        example_4_react_agent()
        example_5_dataset_catalog()
        example_6_state_inspection()

        # Uncomment to run full agent (requires paper and API keys)
        # example_1_basic_usage()

        print("\n" + "=" * 80)
        print("✅ All examples completed!")
        print("=" * 80)
        print("\nTo run the full agent:")
        print("  python main.py --paper /path/to/paper.pdf")
        print("\nOr:")
        print("  python example.py  # Uncomment example_1_basic_usage()")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
