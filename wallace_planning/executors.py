"""Fine-grained plan execution using Wallace's existing tool registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .generators import save_plan
from .models import MAX_PLAN_RESULT_CHARS, PlanSpec, PlanStep, bounded, utc_now


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", None) or getattr(tool, "__name__", "tool"))


def _run_tool(tool: Any, args: Mapping[str, Any]) -> Any:
    if hasattr(tool, "invoke"):
        return tool.invoke(dict(args))
    if callable(tool):
        return tool(**dict(args))
    raise TypeError(f"registered tool {_tool_name(tool)} is not callable")


class PlanExecutor:
    def __init__(
        self,
        project_root: Path,
        tool_registry: list[Any] | Mapping[str, Any] | None = None,
        approval_required: set[str] | None = None,
    ) -> None:
        self.project_root = project_root
        if approval_required is None:
            try:
                from tools import TOOLS_REQUIRING_APPROVAL

                approval_required = set(TOOLS_REQUIRING_APPROVAL)
            except ImportError:
                approval_required = set()
        self.approval_required = approval_required
        if isinstance(tool_registry, Mapping):
            self.tools = dict(tool_registry)
        else:
            self.tools = {_tool_name(tool): tool for tool in (tool_registry or [])}

    def execute_next(
        self,
        plan: PlanSpec,
        *,
        approval_callback: Callable[[PlanStep], bool] | None = None,
        tool_runner: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> PlanStep | None:
        step = plan.next_pending()
        if step is None:
            plan.refresh_status()
            save_plan(self.project_root, plan)
            return None

        step.status = "running"
        step.updated_at = utc_now()
        save_plan(self.project_root, plan)
        try:
            if step.tool_name:
                if step.tool_name not in self.tools:
                    raise ValueError(f"tool is not registered: {step.tool_name}")
                if step.tool_name in self.approval_required and approval_callback is None:
                    raise PermissionError(f"approval callback required for {step.tool_name}")
                if approval_callback is not None and not approval_callback(step):
                    raise PermissionError(f"approval rejected for {step.tool_name}")
                result = (
                    tool_runner(step.tool_name, dict(step.tool_args))
                    if tool_runner is not None
                    else _run_tool(self.tools[step.tool_name], step.tool_args)
                )
                step.result = bounded(result, MAX_PLAN_RESULT_CHARS)
            else:
                step.result = bounded(
                    "No direct tool call recorded; the model should complete this reasoning step and summarize it.",
                    MAX_PLAN_RESULT_CHARS,
                )
            step.status = "complete"
            step.updated_at = utc_now()
            plan.last_error = ""
        except Exception as error:
            step.status = "failed"
            step.result = bounded(f"{type(error).__name__}: {error}", MAX_PLAN_RESULT_CHARS)
            step.updated_at = utc_now()
            plan.last_error = step.result
            plan.refresh_status()
            save_plan(self.project_root, plan)
            return step

        plan.refresh_status()
        save_plan(self.project_root, plan)
        return step

    def execute_all(
        self,
        plan: PlanSpec,
        *,
        approval_callback: Callable[[PlanStep], bool] | None = None,
        tool_runner: Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> PlanSpec:
        while plan.next_pending() is not None:
            step = self.execute_next(plan, approval_callback=approval_callback, tool_runner=tool_runner)
            if step is None or step.status == "failed":
                break
        plan.refresh_status()
        save_plan(self.project_root, plan)
        return plan
