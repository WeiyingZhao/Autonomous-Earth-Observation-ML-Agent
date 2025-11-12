#!/usr/bin/env python3
"""
Main orchestrator for the ML Reproduction Agent.
Entry point for running paper reproduction end-to-end.
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.graph import create_ml_agent_graph
from src.agent.state import AgentState, init_phase_tracking
from src.agent.router import init_router


def setup_environment():
    """
    Setup environment and load API keys.
    """
    # Load .env file
    load_dotenv()

    # Check for required API keys
    required_keys = ["OPENAI_API_KEY", "DEEPSEEK_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print(f"⚠ Warning: Missing API keys: {', '.join(missing_keys)}")
        print("The agent may fail if it needs these providers.")
        print("Set them in .env or as environment variables.")

    # Initialize router
    init_router()


def run_agent(
    paper_uri: str,
    task_hint: str = None,
    max_gpu_hours: float = 6.0,
    target_sensors: list = None,
    output_dir: str = None
):
    """
    Run the ML reproduction agent on a paper.

    Args:
        paper_uri: Path to PDF or arXiv link
        task_hint: Optional hint about the task (classification, segmentation, etc.)
        max_gpu_hours: Maximum GPU hours for training
        target_sensors: Optional list of target sensors
        output_dir: Optional custom output directory

    Returns:
        Final agent state
    """
    print("=" * 80)
    print("ML REPRODUCTION AGENT")
    print("=" * 80)
    print(f"Paper: {paper_uri}")
    print(f"Task hint: {task_hint or 'Auto-detect'}")
    print(f"Max GPU hours: {max_gpu_hours}")
    print(f"Target sensors: {target_sensors or 'Auto-detect'}")
    print("=" * 80)

    # Create agent graph
    graph = create_ml_agent_graph(use_checkpointer=True)

    # Initialize state
    artifacts_dir = output_dir or f"artifacts/run_{os.path.basename(paper_uri).replace('.pdf', '')}"
    os.makedirs(artifacts_dir, exist_ok=True)

    state = AgentState(
        paper_uri=paper_uri,
        task_hint=task_hint,
        max_gpu_hours=max_gpu_hours,
        target_sensors=target_sensors or [],
        artifacts_dir=artifacts_dir
    )

    # Initialize phase tracking
    state = init_phase_tracking(state)

    # Run agent
    try:
        print("\n🚀 Starting agent execution...\n")
        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": state.run_id}}
        )

        print("\n" + "=" * 80)
        print("EXECUTION COMPLETE")
        print("=" * 80)

        # Print summary
        print("\n📊 Summary:")

        # Handle dict vs AgentState return type (LangGraph returns dict)
        if isinstance(result, dict):
            run_id = result.get('run_id', 'N/A')
            paper_spec = result.get('paper_spec')
            dataset_info = result.get('dataset_info')
            code_artifacts = result.get('code_artifacts')
            training_results = result.get('training_results')
            report_info = result.get('report_info')
            artifacts_dir = result.get('artifacts_dir', 'N/A')
            phases = result.get('phases', {})
            errors = result.get('errors', [])
        else:
            run_id = result.run_id
            paper_spec = result.paper_spec
            dataset_info = result.dataset_info
            code_artifacts = result.code_artifacts
            training_results = result.training_results
            report_info = result.report_info
            artifacts_dir = result.artifacts_dir
            phases = result.phases
            errors = result.errors

        print(f"  Run ID: {run_id}")
        print(f"  Paper: {paper_spec.title if paper_spec else 'N/A'}")
        print(f"  Dataset: {dataset_info.name if dataset_info else 'N/A'}")
        print(f"  Model: {code_artifacts.model_architecture if code_artifacts else 'N/A'}")

        if training_results:
            print(f"\n📈 Results:")
            metrics = training_results.metrics if hasattr(training_results, 'metrics') else training_results.get('metrics', {})
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        if report_info:
            report_path = report_info.report_path if hasattr(report_info, 'report_path') else report_info.get('report_path')
            print(f"\n📄 Report: {report_path}")

        print(f"\n📁 Artifacts: {artifacts_dir}")

        # Print phase status
        print(f"\n✅ Phase Status:")
        for phase_name, phase in phases.items():
            status_value = phase.status.value if hasattr(phase, 'status') else phase.get('status')
            status_icon = {
                "completed": "✓",
                "failed": "✗",
                "in_progress": "⟳",
                "pending": "○"
            }.get(status_value, "?")

            print(f"  {status_icon} {phase_name}: {status_value}")

        # Print errors if any
        if errors:
            print(f"\n⚠ Errors ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")

        return result

    except Exception as e:
        print(f"\n❌ Agent execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main entry point with CLI argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="ML Reproduction Agent - Automatically reproduce ML papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on a local PDF
  python main.py --paper /path/to/paper.pdf

  # Run on arXiv paper
  python main.py --paper arxiv:2103.14030 --task segmentation

  # Custom settings
  python main.py --paper paper.pdf --task classification --gpu-hours 4 --sensors Sentinel-2

  # Specify output directory
  python main.py --paper paper.pdf --output-dir ./my_reproduction
        """
    )

    parser.add_argument(
        "--paper",
        type=str,
        required=True,
        help="Path to paper PDF or arXiv ID (e.g., arxiv:2103.14030)"
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "segmentation", "detection", "regression"],
        help="Task type hint (auto-detected if not specified)"
    )

    parser.add_argument(
        "--gpu-hours",
        type=float,
        default=6.0,
        help="Maximum GPU hours for training (default: 6.0)"
    )

    parser.add_argument(
        "--sensors",
        type=str,
        nargs="+",
        help="Target sensors (e.g., Sentinel-2, SAR)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory for artifacts"
    )

    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip training phase (generate code only)"
    )

    args = parser.parse_args()

    # Setup environment
    setup_environment()

    # Run agent
    result = run_agent(
        paper_uri=args.paper,
        task_hint=args.task,
        max_gpu_hours=args.gpu_hours,
        target_sensors=args.sensors,
        output_dir=args.output_dir
    )

    # Exit with appropriate code
    # Handle dict vs AgentState return type
    report_info = result.get('report_info') if isinstance(result, dict) else (result.report_info if result else None)

    if result and report_info:
        print("\n✅ SUCCESS: Reproduction complete!")
        sys.exit(0)
    else:
        print("\n❌ FAILED: Reproduction incomplete")
        sys.exit(1)


if __name__ == "__main__":
    main()
