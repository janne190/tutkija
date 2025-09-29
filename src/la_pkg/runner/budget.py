# SPDX-FileCopyrightText: 2024-present tutkija <tutkija@tutkija.fi>
#
# SPDX-License-Identifier: MIT

from datetime import datetime
from typing import Dict, Any, Optional

from la_pkg.runner.schemas import NodeMetrics, RunState


class BudgetTracker:
    """Tracks LLM costs and execution times for a research run."""

    def __init__(self):
        self._current_metrics: Dict[str, NodeMetrics] = {}

    def start_node(self, node_name: str):
        """Records the start time for a node."""
        self._current_metrics[node_name] = NodeMetrics(
            node_name=node_name, start_time=datetime.now(), end_time=datetime.now(), duration_s=0.0
        )

    def end_node(
        self,
        node_name: str,
        status: str = "success",
        error_message: Optional[str] = None,
        llm_cost_usd: float = 0.0,
        token_count: int = 0,
    ):
        """Records the end time and metrics for a node."""
        if node_name not in self._current_metrics:
            raise ValueError(f"Node '{node_name}' was not started.")

        metrics = self._current_metrics[node_name]
        metrics.end_time = datetime.now()
        metrics.duration_s = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.status = status
        metrics.error_message = error_message
        metrics.llm_cost_usd = llm_cost_usd
        metrics.token_count = token_count
        return metrics

    def update_run_state(self, state: RunState, node_name: str, llm_cost_usd: float = 0.0, token_count: int = 0, status: str = "success", error_message: Optional[str] = None) -> RunState:
        """Updates the overall run state with node metrics and budget info."""
        metrics = self.end_node(node_name, llm_cost_usd=llm_cost_usd, token_count=token_count, status=status, error_message=error_message)
        state.metrics.append(metrics)
        state.total_llm_cost_usd += metrics.llm_cost_usd
        state.total_duration_s += metrics.duration_s
        return state

    def check_budget(self, state: RunState) -> bool:
        """Checks if the run is within the allocated budget."""
        return state.total_llm_cost_usd <= state.config.budget_usd

    def check_iterations(self, state: RunState) -> bool:
        """Checks if the run is within the maximum allowed iterations."""
        return state.iteration < state.config.max_iterations
