"""
Task definitions for the Customer Support Environment.

Each task represents a realistic support scenario with:
- A customer ticket
- Incremental step rewards for relevant tool calls
- A deterministic grader that scores the final resolution 0.0 → 1.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    score: float          # 0.0 to 1.0
    breakdown: Dict[str, float]
    feedback: str


# ---------------------------------------------------------------------------
# Base Task
# ---------------------------------------------------------------------------

class Task:
    name: str
    difficulty: str
    ticket_id: str
    ticket_content: str
    max_steps: int

    def reward_for_kb_lookup(self, matched_topic: str, lookups_done: Set[str]) -> float:
        """Incremental reward when the agent queries the knowledge base."""
        raise NotImplementedError

    def reward_for_data_lookup(
        self, lookup_type: str, id_value: str, lookups_done: Set[str]
    ) -> float:
        """Incremental reward for order / account / billing lookups."""
        raise NotImplementedError

    def grade(
        self,
        lookups_done: Set[str],
        resolution_type: str,
        message: str,
        is_escalation: bool,
    ) -> TaskResult:
        """Score the completed episode deterministically."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Task 1 — Easy: Refund for Lost Package
# ---------------------------------------------------------------------------

class EasyRefundTask(Task):
    """
    Customer ordered Wireless Headphones Pro (#ORD-1001).
    The package was lost in transit. They want a refund.

    Optimal path  (3–4 steps):
        lookup_order(ORD-1001)
        [optional] lookup_knowledge_base("refund policy")
        submit_resolution(resolution_type="refund", message=<...>)
    """

    name = "easy_refund"
    difficulty = "easy"
    ticket_id = "TKT-0001"
    ticket_content = (
        "Hi, I placed an order for Wireless Headphones Pro on March 11th "
        "(Order #ORD-1001, total $45.99). It's now been over three weeks and "
        "I still haven't received my package. The tracking page shows it was "
        "lost in transit. I'd really like a full refund. Please help!"
        "\n\n— Sarah M."
    )
    max_steps = 6

    def reward_for_kb_lookup(self, matched_topic: str, lookups_done: Set[str]) -> float:
        if matched_topic == "refund" and "kb:refund" not in lookups_done:
            return 0.10
        return 0.0

    def reward_for_data_lookup(
        self, lookup_type: str, id_value: str, lookups_done: Set[str]
    ) -> float:
        key = f"{lookup_type}:{id_value}"
        if key in lookups_done:
            return -0.10  # penalty: repeated lookup (loop prevention)
        if lookup_type == "order" and id_value == "ORD-1001":
            return 0.30   # correct, relevant lookup
        if lookup_type == "order":
            return -0.05  # wrong order ID
        return 0.0        # irrelevant lookup type

    def grade(
        self,
        lookups_done: Set[str],
        resolution_type: str,
        message: str,
        is_escalation: bool,
    ) -> TaskResult:
        score = 0.0
        breakdown: Dict[str, float] = {}
        msg = message.lower()

        # 1. Looked up the correct order (0.35)
        found_order = "order:ORD-1001" in lookups_done
        breakdown["looked_up_correct_order"] = 0.35 if found_order else 0.0
        score += breakdown["looked_up_correct_order"]

        # 2. Correct resolution type: refund (0.35)
        correct_type = resolution_type == "refund"
        breakdown["correct_resolution_type"] = 0.35 if correct_type else 0.0
        score += breakdown["correct_resolution_type"]

        # 3. Message mentions the refund amount (0.15)
        mentions_amount = "45.99" in message or "45" in message
        breakdown["mentions_refund_amount"] = 0.15 if mentions_amount else 0.0
        score += breakdown["mentions_refund_amount"]

        # 4. Message quality: apology and/or processing timeline (0.15)
        has_apology = any(w in msg for w in ["sorry", "apologize", "apologies", "regret"])
        has_timeline = any(w in msg for w in ["3-5", "5 business", "business day", "days"])
        breakdown["message_quality"] = 0.15 if (has_apology or has_timeline) else 0.0
        score += breakdown["message_quality"]

        # Penalty: escalating a straightforward <$100 refund
        if is_escalation:
            score = min(score, 0.25)

        feedback = _build_feedback(
            [
                (not found_order, "Look up order ORD-1001 to verify refund eligibility."),
                (not correct_type, "Use resolution_type='refund' for this scenario."),
                (not mentions_amount, "Include the refund amount ($45.99) in the message."),
            ],
            score,
        )
        return TaskResult(score=round(score, 2), breakdown=breakdown, feedback=feedback)


# ---------------------------------------------------------------------------
# Task 2 — Medium: Account Locked After Suspicious Login
# ---------------------------------------------------------------------------

class MediumAccountLockoutTask(Task):
    """
    Customer cannot log in; account was locked after a suspicious login from
    an unknown IP. They suspect unauthorized access.

    Optimal path  (4–6 steps):
        lookup_account(ACC-2047)
        lookup_knowledge_base("account security")
        submit_resolution(resolution_type="account_fix", message=<...>)
        OR escalate(message=<...>)   ← also acceptable
    """

    name = "medium_account_lockout"
    difficulty = "medium"
    ticket_id = "TKT-0002"
    ticket_content = (
        "I'm unable to access my account (Account ID: ACC-2047). I've tried "
        "resetting my password several times but it keeps failing. I also just "
        "received a suspicious login notification from an unknown location — "
        "I'm worried someone has hacked into my account. Please help me regain "
        "access and make sure my account is secure as soon as possible."
        "\n\n— John Smith"
    )
    max_steps = 8

    def reward_for_kb_lookup(self, matched_topic: str, lookups_done: Set[str]) -> float:
        if matched_topic == "account security" and "kb:account security" not in lookups_done:
            return 0.20
        if matched_topic == "escalation" and "kb:escalation" not in lookups_done:
            return 0.05
        return 0.0

    def reward_for_data_lookup(
        self, lookup_type: str, id_value: str, lookups_done: Set[str]
    ) -> float:
        key = f"{lookup_type}:{id_value}"
        if key in lookups_done:
            return -0.10
        if lookup_type == "account" and id_value == "ACC-2047":
            return 0.25
        if lookup_type == "account":
            return -0.05
        return 0.0

    def grade(
        self,
        lookups_done: Set[str],
        resolution_type: str,
        message: str,
        is_escalation: bool,
    ) -> TaskResult:
        score = 0.0
        breakdown: Dict[str, float] = {}
        msg = message.lower()

        # 1. Looked up correct account (0.25)
        found_account = "account:ACC-2047" in lookups_done
        breakdown["looked_up_account"] = 0.25 if found_account else 0.0
        score += breakdown["looked_up_account"]

        # 2. Checked security policy in KB (0.20)
        checked_security = "kb:account security" in lookups_done
        breakdown["checked_security_policy"] = 0.20 if checked_security else 0.0
        score += breakdown["checked_security_policy"]

        # 3. Correct resolution type: account_fix or escalate (0.25)
        valid_types = {"account_fix", "escalate"}
        correct_type = resolution_type in valid_types or is_escalation
        breakdown["correct_resolution_type"] = 0.25 if correct_type else 0.0
        score += breakdown["correct_resolution_type"]

        # 4. Message addresses security concern (0.15)
        mentions_security = any(
            w in msg for w in ["security", "suspicious", "secure", "protect", "unauthoriz"]
        )
        breakdown["mentions_security"] = 0.15 if mentions_security else 0.0
        score += breakdown["mentions_security"]

        # 5. Message mentions verification or next steps (0.15)
        mentions_next_steps = any(
            w in msg for w in ["verif", "identity", "email", "24", "48", "hour", "review", "unlock"]
        )
        breakdown["mentions_next_steps"] = 0.15 if mentions_next_steps else 0.0
        score += breakdown["mentions_next_steps"]

        feedback = _build_feedback(
            [
                (not found_account, "Look up account ACC-2047 to understand why it was locked."),
                (not checked_security, "Check the knowledge base for account security policy."),
                (not correct_type, "Use resolution_type='account_fix' or 'escalate' for security incidents."),
                (not mentions_security, "Address the security concern explicitly in your message."),
                (not mentions_next_steps, "Tell the customer what happens next (verification, timeline)."),
            ],
            score,
        )
        return TaskResult(score=round(score, 2), breakdown=breakdown, feedback=feedback)


# ---------------------------------------------------------------------------
# Task 3 — Hard: Multi-Issue Billing Dispute
# ---------------------------------------------------------------------------

class HardBillingDisputeTask(Task):
    """
    Customer has three concurrent billing issues:
      (1) Charged $89.99 after subscription cancellation (#ORD-3055) — INVALID
      (2) Unrecognised $12.50 charge — VALID (API usage)
      (3) Missing $20 loyalty credit on account ACC-5091

    Optimal path  (5–9 steps):
        lookup_order(ORD-3055)
        lookup_billing(ACC-5091)
        lookup_account(ACC-5091)
        [optional KB lookups for billing / credits policy]
        submit_resolution covering all three issues
    """

    name = "hard_billing_dispute"
    difficulty = "hard"
    ticket_id = "TKT-0003"
    ticket_content = (
        "I have several billing problems I need resolved urgently:\n\n"
        "1. I cancelled my Pro Subscription (#ORD-3055) on March 10th, but I was "
        "still charged $89.99 on March 15th. This is completely wrong and I want "
        "a refund.\n\n"
        "2. I also see a charge of $12.50 from March 1st that I do not recognise "
        "at all. What is this for?\n\n"
        "3. Back in February your support team promised me a $20 credit for the "
        "service outage. It's now April and that credit has still not appeared on "
        "my account (ACC-5091). Please sort all of this out."
        "\n\n— Maria Wilson"
    )
    max_steps = 12

    def reward_for_kb_lookup(self, matched_topic: str, lookups_done: Set[str]) -> float:
        rewards = {
            "billing": 0.05,
            "subscription": 0.05,
            "credits": 0.05,
        }
        key = f"kb:{matched_topic}"
        if matched_topic in rewards and key not in lookups_done:
            return rewards[matched_topic]
        return 0.0

    def reward_for_data_lookup(
        self, lookup_type: str, id_value: str, lookups_done: Set[str]
    ) -> float:
        key = f"{lookup_type}:{id_value}"
        if key in lookups_done:
            return -0.10
        relevant = {
            ("order", "ORD-3055"): 0.15,
            ("billing", "ACC-5091"): 0.20,
            ("account", "ACC-5091"): 0.15,
        }
        return relevant.get((lookup_type, id_value), -0.05)

    def grade(
        self,
        lookups_done: Set[str],
        resolution_type: str,
        message: str,
        is_escalation: bool,
    ) -> TaskResult:
        score = 0.0
        breakdown: Dict[str, float] = {}
        msg = message.lower()

        # 1. Looked up order ORD-3055 (0.12)
        found_order = "order:ORD-3055" in lookups_done
        breakdown["looked_up_order"] = 0.12 if found_order else 0.0
        score += breakdown["looked_up_order"]

        # 2. Looked up billing history (0.12)
        found_billing = "billing:ACC-5091" in lookups_done
        breakdown["looked_up_billing"] = 0.12 if found_billing else 0.0
        score += breakdown["looked_up_billing"]

        # 3. Looked up account (0.11)
        found_account = "account:ACC-5091" in lookups_done
        breakdown["looked_up_account"] = 0.11 if found_account else 0.0
        score += breakdown["looked_up_account"]

        # 4. Handles invalid subscription charge — refund (0.20)
        handles_refund = resolution_type == "refund" or any(
            w in msg for w in ["refund", "89.99", "subscription charge", "invalid charge", "incorrect charge"]
        )
        breakdown["handles_subscription_refund"] = 0.20 if handles_refund else 0.0
        score += breakdown["handles_subscription_refund"]

        # 5. Explains valid $12.50 API usage charge (0.15)
        explains_valid = any(
            w in msg for w in ["12.50", "api", "usage", "valid", "legitimate", "premium api"]
        )
        breakdown["explains_valid_charge"] = 0.15 if explains_valid else 0.0
        score += breakdown["explains_valid_charge"]

        # 6. Addresses missing $20 credit (0.20)
        addresses_credit = any(
            w in msg for w in ["20", "$20", "credit", "applied", "loyalty", "outage", "compensation"]
        )
        breakdown["addresses_missing_credit"] = 0.20 if addresses_credit else 0.0
        score += breakdown["addresses_missing_credit"]

        # 7. Bonus: all three issues addressed in one response (0.10)
        all_addressed = handles_refund and explains_valid and addresses_credit
        breakdown["all_issues_addressed"] = 0.10 if all_addressed else 0.0
        score += breakdown["all_issues_addressed"]

        feedback = _build_feedback(
            [
                (not found_order, "Look up order ORD-3055 to verify the cancellation date."),
                (not found_billing, "Look up billing history for ACC-5091 to identify all charges."),
                (not found_account, "Look up account ACC-5091 to find the pending $20 credit."),
                (not handles_refund, "Must refund the invalid $89.99 subscription charge."),
                (not explains_valid, "Explain that the $12.50 is a valid API usage charge."),
                (not addresses_credit, "Must address and apply the missing $20 loyalty credit."),
            ],
            score,
        )
        return TaskResult(score=round(score, 2), breakdown=breakdown, feedback=feedback)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_tasks() -> Dict[str, Task]:
    return {
        "easy_refund": EasyRefundTask(),
        "medium_account_lockout": MediumAccountLockoutTask(),
        "hard_billing_dispute": HardBillingDisputeTask(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_feedback(checks: List[tuple], score: float) -> str:
    issues = [msg for failed, msg in checks if failed]
    if score >= 0.85:
        return "Excellent resolution! " + " ".join(issues)
    if score >= 0.60:
        return "Good resolution with minor gaps. " + " ".join(issues)
    return "Needs improvement. " + " ".join(issues)
