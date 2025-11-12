"""
LangGraph construction for the ML reproduction agent.
Defines the state machine with nodes, edges, and conditional routing.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.agent.state import AgentState, init_phase_tracking
from src.agent.nodes import (
    parse_paper_node,
    validate_spec_node,
    resolve_dataset_node,
    synthesize_code_node,
    prepare_data_node,
    train_evaluate_node,
    generate_report_node,
    should_retry
)


def create_ml_agent_graph(use_checkpointer: bool = True) -> StateGraph:
    """
    Create the LangGraph for the ML reproduction agent.

    Graph Structure (from readme.md):
    START → ParsePaper → ValidateSpec → ResolveDataset →
    SynthesizeCode → PrepareData → TrainEvaluate → GenerateReport → END

    Args:
        use_checkpointer: Whether to use MemorySaver for state persistence

    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes (phases)
    workflow.add_node("parse_paper", parse_paper_node)
    workflow.add_node("validate_spec", validate_spec_node)
    workflow.add_node("resolve_dataset", resolve_dataset_node)
    workflow.add_node("synthesize_code", synthesize_code_node)
    workflow.add_node("prepare_data", prepare_data_node)
    workflow.add_node("train_evaluate", train_evaluate_node)
    workflow.add_node("generate_report", generate_report_node)

    # Set entry point
    workflow.set_entry_point("parse_paper")

    # Add edges (sequential flow)
    workflow.add_edge("parse_paper", "validate_spec")
    workflow.add_edge("validate_spec", "resolve_dataset")
    workflow.add_edge("resolve_dataset", "synthesize_code")
    workflow.add_edge("synthesize_code", "prepare_data")
    workflow.add_edge("prepare_data", "train_evaluate")
    workflow.add_edge("train_evaluate", "generate_report")

    # Conditional edge: retry or end
    workflow.add_conditional_edges(
        "generate_report",
        should_retry,
        {
            "retry": "parse_paper",  # Start over if retry needed
            "end": END
        }
    )

    # Add checkpointer for state persistence
    if use_checkpointer:
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()


def create_agent_with_tools():
    """
    Alternative: Create a ReAct agent with tools.
    This is a simpler approach using LangChain's prebuilt agents.

    Returns:
        Configured agent executor
    """
    from langgraph.prebuilt import create_react_agent
    from src.agent.router import get_router
    from src.tools import (
        ingest_paper_tool,
        resolve_dataset_tool,
        synthesize_code_tool
    )

    # Get LLM
    router = get_router()
    llm = router.get_model("general")

    # Create tools list
    tools = [
        ingest_paper_tool,
        resolve_dataset_tool,
        synthesize_code_tool
    ]

    # Create agent
    agent = create_react_agent(
        llm,
        tools,
        checkpointer=MemorySaver()
    )

    return agent


# Example usage
if __name__ == "__main__":
    # Create graph
    graph = create_ml_agent_graph()

    # Initialize state
    from src.agent.state import AgentState
    import os

    state = AgentState(
        paper_uri="/path/to/paper.pdf",
        task_hint="segmentation",
        max_gpu_hours=4.0
    )

    # Initialize phase tracking
    state = init_phase_tracking(state)

    # Run graph
    print("=" * 60)
    print("ML Reproduction Agent - Starting")
    print("=" * 60)

    try:
        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": state.run_id}}
        )

        print("\n" + "=" * 60)
        print("Execution Complete")
        print("=" * 60)

        if result.report_info:
            print(f"\nReport generated: {result.report_info.report_path}")
            print(f"Artifacts directory: {result.artifacts_dir}")

        if result.errors:
            print(f"\nErrors encountered: {len(result.errors)}")
            for error in result.errors:
                print(f"  - {error}")

    except Exception as e:
        print(f"\n✗ Agent execution failed: {str(e)}")
