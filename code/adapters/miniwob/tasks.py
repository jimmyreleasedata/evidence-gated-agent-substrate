"""Task inventory for the MiniWoB++ v1 slice."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MiniWobTaskSpec:
    task_id: str
    category: str
    default_action: str
    step_delay_ms: int
    instruction: str


V1_TASKS = [
    MiniWobTaskSpec("click-button", "click", "click:primary", 6, "Click the primary button."),
    MiniWobTaskSpec("click-checkboxes", "click", "toggle:checkboxes", 7, "Select the requested checkboxes."),
    MiniWobTaskSpec("click-dialog", "click", "click:dialog-confirm", 8, "Dismiss the dialog correctly."),
    MiniWobTaskSpec("click-link", "click", "click:link", 5, "Click the matching link."),
    MiniWobTaskSpec("click-option", "select", "select:option", 7, "Choose the requested option."),
    MiniWobTaskSpec("click-scroll-list", "select", "scroll-select:item", 9, "Scroll and choose the item."),
    MiniWobTaskSpec("enter-date", "text", "type:date", 8, "Enter the target date."),
    MiniWobTaskSpec("enter-text", "text", "type:text", 6, "Type the requested text."),
    MiniWobTaskSpec("focus-text", "text", "focus:textbox", 4, "Move focus to the active textbox."),
    MiniWobTaskSpec("login-user", "form", "submit:login", 10, "Fill credentials and sign in."),
    MiniWobTaskSpec("multi-layouts", "layout", "click:layout-target", 11, "Act under varying page layouts."),
    MiniWobTaskSpec("use-slider", "widget", "drag:slider", 9, "Set the slider to the requested value."),
]

TASKS_BY_ID = {task.task_id: task for task in V1_TASKS}


def get_task(task_id: str) -> MiniWobTaskSpec:
    try:
        return TASKS_BY_ID[task_id]
    except KeyError as exc:
        raise KeyError(f"unknown MiniWoB task_id: {task_id}") from exc


def default_task_ids() -> list[str]:
    return [task.task_id for task in V1_TASKS]
