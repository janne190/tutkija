# SPDX-FileCopyrightText: 2024-present tutkija <tutkija@tutkija.fi>
#
# SPDX-License-Identifier: MIT

from typing import Callable, Dict, List, Literal, Optional, Tuple

from la_pkg.runner.budget import BudgetTracker
from la_pkg.runner.nodes import (
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
from la_pkg.runner.schemas import RunConfig, RunState


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
            state = self.budget_tracker.update_run_state(state, node_name, status="success")
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
            # Check skip flags
            if current_node == "search" and self.config.skip_search:
                state.warnings.append("Skipping search node as requested.")
                current_node = "screen"
                continue
            if current_node == "screen" and self.config.skip_screen:
                state.warnings.append("Skipping screen node as requested.")
                current_node = "ingest"
                continue
            if current_node == "ingest" and self.config.skip_ingest:
                state.warnings.append("Skipping ingest node as requested.")
                current_node = "index"
                continue
            if current_node == "index" and self.config.skip_index:
                state.warnings.append("Skipping index node as requested.")
                current_node = "qa"
                continue
            if current_node == "qa" and self.config.skip_qa:
                state.warnings.append("Skipping QA node as requested.")
                current_node = "write"
                continue
            if current_node == "write" and self.config.skip_write:
                state.warnings.append("Skipping write node as requested.")
                current_node = "prisma"
                continue
            if current_node == "prisma" and self.config.skip_prisma:
                state.warnings.append("Skipping PRISMA node as requested.")
                current_node = "critic"
                continue
            if current_node == "critic" and self.config.skip_critic:
                state.warnings.append("Skipping critic node as requested.")
                current_node = "finalize"
                continue

            if not self.budget_tracker.check_budget(state):
                state.errors.append("Run stopped: Budget exceeded.")
                state.final_status = "failed"
                break
            if not self.budget_tracker.check_iterations(state):
                state.errors.append("Run stopped: Maximum iterations reached.")
                state.final_status = "failed"
                break

            try:
                state = self._execute_node(current_node, state)
            except Exception:
                # If a node fails, we stop the execution for this simplified DAG runner
                state.errors.append(f"Pipeline stopped due to failure in node: {current_node}")
                state.final_status = "failed" # Set final_status on early exit
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
                    state.final_status = "failed"
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
