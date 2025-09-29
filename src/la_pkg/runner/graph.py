# SPDX-FileCopyrightText: 2024-present tutkija <tutkija@tutkija.fi>
#
# SPDX-License-Identifier: MIT

from typing import Callable, Dict, List, Literal, Tuple

from src.la_pkg.runner.budget import BudgetTracker
from src.la_pkg.runner.nodes import (
    critic_node,
    finalize_node,
    index_node,
    ingest_node,
    plan_node,
    prisma_node,
    qa_node,
    screen_node,
    search_node,
    write_node,
)
from src.la_pkg.runner.schemas import RunConfig, RunState


class DAGRunner:
    """
    A lightweight Directed Acyclic Graph (DAG) runner for orchestrating research nodes.
    This serves as a fallback when LangGraph is not used.
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.budget_tracker = BudgetTracker()
        self.nodes: Dict[str, Callable[[RunState], RunState]] = {
            "plan": plan_node,
            "search": search_node,
            "screen": screen_node,
            "ingest": ingest_node,
            "index": index_node,
            "qa": qa_node,
            "write": write_node,
            "prisma": prisma_node,
            "critic": critic_node,
            "finalize": finalize_node,
        }
        self.graph: Dict[str, List[str]] = {
            "plan": ["search"],
            "search": ["screen"],
            "screen": ["ingest"],
            "ingest": ["index"],
            "index": ["qa"],
            "qa": ["write"],
            "write": ["prisma"],
            "prisma": ["critic"],
            "critic": ["finalize"],  # Simplified for initial DAG, critic feedback loop handled in run()
            "finalize": [],
        }

    def _execute_node(self, node_name: str, state: RunState) -> RunState:
        """Executes a single node and updates the state and metrics."""
        self.budget_tracker.start_node(node_name)
        try:
            # Special handling for plan_node as it needs config
            if node_name == "plan":
                state = self.nodes[node_name](state, self.config)
            else:
                state = self.nodes[node_name](state)
            state = self.budget_tracker.update_run_state(state, node_name)
        except Exception as e:
            state = self.budget_tracker.update_run_state(state, node_name, status="failed", error_message=str(e))
            state.errors.append(f"Node '{node_name}' failed: {e}")
            raise
        return state

    def run(self, initial_state: Optional[RunState] = None) -> RunState:
        """
        Runs the research pipeline using the DAG.
        Handles critic feedback loop with a simple iteration mechanism.
        """
        state = initial_state if initial_state else RunState(config=self.config, run_id="")
        current_node = "plan"

        while current_node:
            if not self.budget_tracker.check_budget(state):
                state.errors.append("Run stopped: Budget exceeded.")
                break
            if not self.budget_tracker.check_iterations(state):
                state.errors.append("Run stopped: Maximum iterations reached.")
                break

            try:
                state = self._execute_node(current_node, state)
            except Exception:
                # If a node fails, we stop the execution for this simplified DAG runner
                state.errors.append(f"Pipeline stopped due to failure in node: {current_node}")
                break

            if current_node == "critic" and state.critic_report and state.critic_report.overall_status == "fail":
                state.iteration += 1
                if self.budget_tracker.check_iterations(state):
                    # Simplified re-run: go back to QA or Index
                    # For a real LangGraph, this would be a more explicit edge
                    state.warnings.append(f"Critic failed, re-running from QA. Iteration {state.iteration}")
                    current_node = "qa"  # Or "index" if deeper changes are needed
                else:
                    state.errors.append("Critic failed and max iterations reached. Stopping.")
                    current_node = None  # Stop the loop
            else:
                next_nodes = self.graph.get(current_node)
                if next_nodes:
                    current_node = next_nodes[0]  # Simple sequential for DAG fallback
                else:
                    current_node = None  # End of graph

        # Ensure finalize runs even if there were errors
        if current_node != "finalize" and "finalize" in self.nodes and not any(m.node_name == "finalize" for m in state.metrics):
            try:
                state = self._execute_node("finalize", state)
            except Exception as e:
                state.errors.append(f"Finalize node failed during error handling: {e}")

        return state
