from typing import Any, Dict, List, Literal, Optional
from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


class SupportAction(Action):
    """An action the agent can take in the customer support environment."""

    action_type: Literal[
        "lookup_knowledge_base",
        "lookup_order",
        "lookup_account",
        "lookup_billing",
        "submit_resolution",
        "escalate",
    ] = Field(..., description="The type of support action to perform")

    query: Optional[str] = Field(
        None, description="Free-text search query for knowledge base lookup"
    )
    order_id: Optional[str] = Field(
        None, description="Order ID (e.g. ORD-1001) for order lookup"
    )
    account_id: Optional[str] = Field(
        None, description="Account ID (e.g. ACC-2047) for account or billing lookup"
    )
    resolution_type: Optional[
        Literal["refund", "credit", "account_fix", "explanation", "escalate"]
    ] = Field(None, description="Category of resolution being applied")
    message: Optional[str] = Field(
        None, description="Message to customer or escalation notes"
    )


class SupportObservation(Observation):
    """Observation returned by the customer support environment after each step."""

    ticket_id: str = Field(..., description="Unique identifier for the support ticket")
    ticket_content: str = Field(..., description="The customer's original ticket text")
    tool_result: Optional[str] = Field(
        None, description="Output from the last tool call (lookup result or resolution confirmation)"
    )
    reward: float = Field(0.0, description="Reward signal for the last action taken")
    done: bool = Field(False, description="True when the episode has ended")
    success: bool = Field(False, description="True when the task was resolved correctly")
    error_message: Optional[str] = Field(
        None, description="Error description if the last action was invalid"
    )
    step_count: int = Field(0, description="Number of steps taken so far")
    max_steps: int = Field(10, description="Maximum allowed steps for this task")
    task_name: str = Field("", description="Name of the current task")
    task_difficulty: str = Field("", description="Difficulty level: easy / medium / hard")


class SupportState(State):
    """Server-side state for the customer support environment."""

    task_name: str = Field("", description="Current task name")
    task_difficulty: str = Field("", description="Task difficulty level")
    is_done: bool = Field(False, description="Whether the current episode has ended")


__all__ = ["SupportAction", "SupportObservation", "SupportState"]
