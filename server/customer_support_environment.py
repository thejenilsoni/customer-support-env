"""
CustomerSupportEnvironment — OpenEnv-compliant RL environment that simulates
a customer support agent resolving realistic support tickets.

Three tasks:
  easy_refund              — refund a lost package order
  medium_account_lockout   — resolve a locked account with suspicious login
  hard_billing_dispute     — multi-issue billing dispute (3 concurrent issues)
"""

from __future__ import annotations

import uuid
from typing import Optional, Set

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    from openenv.core.env_server.interfaces import Environment

from .knowledge_base import (
    ACCOUNTS,
    BILLING_HISTORY,
    KB_TOPIC_KEYWORDS,
    KNOWLEDGE_BASE,
    ORDERS,
)
from .tasks import Task, build_tasks

try:
    from models import SupportAction, SupportObservation, SupportState
except ImportError:
    from ..models import SupportAction, SupportObservation, SupportState


class CustomerSupportEnvironment(
    Environment[SupportAction, SupportObservation, SupportState]
):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, transform=None, rubric=None):
        super().__init__(transform=transform, rubric=rubric)
        self._tasks = build_tasks()
        self._task: Optional[Task] = None
        self._episode_id: Optional[str] = None
        self._step_count: int = 0
        self._lookups_done: Set[str] = set()
        self._done: bool = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ) -> SupportObservation:
        task_key = task_name if task_name in self._tasks else "easy_refund"
        self._task = self._tasks[task_key]
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._lookups_done = set()
        self._done = False

        return SupportObservation(
            ticket_id=self._task.ticket_id,
            ticket_content=self._task.ticket_content,
            tool_result=None,
            reward=0.01,
            done=False,
            success=False,
            error_message=None,
            step_count=0,
            max_steps=self._task.max_steps,
            task_name=self._task.name,
            task_difficulty=self._task.difficulty,
        )

    def step(
        self,
        action: SupportAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> SupportObservation:
        if self._task is None:
            raise RuntimeError("Call reset() before step().")

        if self._done:
            return SupportObservation(
                ticket_id=self._task.ticket_id,
                ticket_content=self._task.ticket_content,
                tool_result="Episode has already ended. Call reset() to start a new episode.",
                reward=0.01,
                done=True,
                success=False,
                error_message="Episode already done.",
                step_count=self._step_count,
                max_steps=self._task.max_steps,
                task_name=self._task.name,
                task_difficulty=self._task.difficulty,
            )

        self._step_count += 1
        tool_result, reward, done, success, error = self._execute_action(action)

        # Timeout: force-end if max steps hit without resolution
        if self._step_count >= self._task.max_steps and not done:
            done = True
            reward -= 0.10
            tool_result = (tool_result or "") + "\n[TIMEOUT: max steps reached without resolution]"

        self._done = done

        # All rewards must be strictly within open interval (0, 1)
        reward = max(0.01, min(0.99, reward))

        return SupportObservation(
            ticket_id=self._task.ticket_id,
            ticket_content=self._task.ticket_content,
            tool_result=tool_result,
            reward=reward,
            done=done,
            success=success,
            error_message=error,
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            task_name=self._task.name,
            task_difficulty=self._task.difficulty,
        )

    @property
    def state(self) -> SupportState:
        return SupportState(
            episode_id=self._episode_id or "",
            step_count=self._step_count,
            task_name=self._task.name if self._task else "",
            task_difficulty=self._task.difficulty if self._task else "",
            is_done=self._done,
        )

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(
        self, action: SupportAction
    ) -> tuple[Optional[str], float, bool, bool, Optional[str]]:
        """
        Returns (tool_result, reward, done, success, error_message).
        """
        atype = action.action_type

        # ---- Knowledge base lookup ----
        if atype == "lookup_knowledge_base":
            return self._handle_kb_lookup(action.query or "")

        # ---- Order lookup ----
        if atype == "lookup_order":
            return self._handle_order_lookup(action.order_id or "")

        # ---- Account lookup ----
        if atype == "lookup_account":
            return self._handle_account_lookup(action.account_id or "")

        # ---- Billing lookup ----
        if atype == "lookup_billing":
            return self._handle_billing_lookup(action.account_id or "")

        # ---- Terminal: submit resolution ----
        if atype == "submit_resolution":
            return self._handle_resolution(
                resolution_type=action.resolution_type or "",
                message=action.message or "",
                is_escalation=False,
            )

        # ---- Terminal: escalate ----
        if atype == "escalate":
            return self._handle_resolution(
                resolution_type="escalate",
                message=action.message or "",
                is_escalation=True,
            )

        return None, -0.10, False, False, f"Unknown action_type: {atype!r}"

    # ------------------------------------------------------------------
    # Individual handlers
    # ------------------------------------------------------------------

    def _handle_kb_lookup(self, query: str):
        q = query.lower()
        matched_topic = None
        for topic, keywords in KB_TOPIC_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                matched_topic = topic
                break

        if matched_topic is None:
            categories = ", ".join(KNOWLEDGE_BASE.keys())
            return (
                f"No article found for '{query}'. "
                f"Available topics: {categories}. Try a more specific query.",
                0.0,
                False,
                False,
                None,
            )

        lookup_key = f"kb:{matched_topic}"
        if lookup_key in self._lookups_done:
            reward = -0.05  # penalty for repeating the same KB lookup
        else:
            reward = self._task.reward_for_kb_lookup(matched_topic, self._lookups_done)
        self._lookups_done.add(lookup_key)

        article = KNOWLEDGE_BASE[matched_topic]
        result = f"[Knowledge Base: {article['title']}]\n{article['content']}"
        return result, reward, False, False, None

    def _handle_order_lookup(self, raw_id: str):
        order_id = raw_id.upper().strip()
        lookup_key = f"order:{order_id}"

        if order_id not in ORDERS:
            return (
                f"Order '{order_id}' not found. Check the order ID and try again.",
                -0.05,
                False,
                False,
                None,
            )

        if lookup_key in self._lookups_done:
            reward = -0.10
        else:
            reward = self._task.reward_for_data_lookup("order", order_id, self._lookups_done)
        self._lookups_done.add(lookup_key)

        order = ORDERS[order_id]
        lines = [f"[Order {order_id}]"] + [f"  {k}: {v}" for k, v in order.items()]
        return "\n".join(lines), reward, False, False, None

    def _handle_account_lookup(self, raw_id: str):
        account_id = raw_id.upper().strip()
        lookup_key = f"account:{account_id}"

        if account_id not in ACCOUNTS:
            return (
                f"Account '{account_id}' not found. Check the account ID and try again.",
                -0.05,
                False,
                False,
                None,
            )

        if lookup_key in self._lookups_done:
            reward = -0.10
        else:
            reward = self._task.reward_for_data_lookup("account", account_id, self._lookups_done)
        self._lookups_done.add(lookup_key)

        account = ACCOUNTS[account_id]
        lines = [f"[Account {account_id}]"] + [f"  {k}: {v}" for k, v in account.items()]
        return "\n".join(lines), reward, False, False, None

    def _handle_billing_lookup(self, raw_id: str):
        account_id = raw_id.upper().strip()
        lookup_key = f"billing:{account_id}"

        if account_id not in BILLING_HISTORY:
            return (
                f"No billing history found for account '{account_id}'.",
                -0.05,
                False,
                False,
                None,
            )

        if lookup_key in self._lookups_done:
            reward = -0.10
        else:
            reward = self._task.reward_for_data_lookup("billing", account_id, self._lookups_done)
        self._lookups_done.add(lookup_key)

        history = BILLING_HISTORY[account_id]
        lines = [f"[Billing History — {account_id}]"]
        for charge in history:
            valid_str = "VALID" if charge["valid"] else "INVALID"
            lines.append(
                f"  {charge['date']}  ${charge['amount']:.2f}  "
                f"{charge['description']}  [{valid_str}]"
            )
            if not charge["valid"]:
                lines.append(f"    ⚠ {charge['explanation']}")
            else:
                lines.append(f"    ✓ {charge['explanation']}")
        return "\n".join(lines), reward, False, False, None

    def _handle_resolution(
        self, resolution_type: str, message: str, is_escalation: bool
    ):
        result = self._task.grade(
            lookups_done=self._lookups_done,
            resolution_type=resolution_type,
            message=message,
            is_escalation=is_escalation,
        )
        # Scores must be strictly within (0, 1) — clamp to open interval
        result.score = max(0.01, min(0.99, result.score))
        success = result.score >= 0.60
        action_word = "escalated" if is_escalation else "submitted"
        tool_result = (
            f"Resolution {action_word}.\n"
            f"Score: {result.score:.2f}/1.00\n"
            f"Feedback: {result.feedback}\n"
            f"Breakdown: {result.breakdown}"
        )
        return tool_result, result.score, True, success, None
