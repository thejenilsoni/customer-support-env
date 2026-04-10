"""
inference.py — Baseline inference script for the Customer Support Environment.

Reads environment variables, runs an LLM agent through all three tasks, and
emits structured output per the OpenEnv hackathon specification.

Required env vars:
    HF_TOKEN      — HuggingFace access token (create at hf.co/settings/tokens)

Optional env vars (with defaults):
    API_BASE_URL  — LLM API endpoint   (default: https://router.huggingface.co/v1)
    MODEL_NAME    — Model identifier    (default: meta-llama/Llama-3.1-8B-Instruct)

Output format (exactly per spec):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys

from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment variables  (API_BASE_URL and MODEL_NAME must have defaults)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN: str | None = os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_SECRET")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ---------------------------------------------------------------------------
# Environment imports (in-process — no HTTP round-trip)
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SupportAction, SupportObservation          # noqa: E402
from server.customer_support_environment import (             # noqa: E402
    CustomerSupportEnvironment,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASKS = ["easy_refund", "medium_account_lockout", "hard_billing_dispute"]
BENCHMARK = "customer_support_env"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert customer support agent. Resolve support tickets efficiently.

Available tools — respond with a single JSON object:

{"action_type": "lookup_knowledge_base", "query": "<terms>"}
{"action_type": "lookup_order",          "order_id": "ORD-XXXX"}
{"action_type": "lookup_account",        "account_id": "ACC-XXXX"}
{"action_type": "lookup_billing",        "account_id": "ACC-XXXX"}
{"action_type": "submit_resolution",     "resolution_type": "refund|credit|account_fix|explanation|escalate", "message": "<customer reply>"}
{"action_type": "escalate",              "message": "<reason>"}

Strategy:
- Gather information first (look up orders, accounts, billing as needed).
- Check the knowledge base for relevant policies.
- Submit a complete, empathetic resolution that addresses all concerns,
  mentions specific amounts/dates, and explains next steps.
- Never repeat the same lookup.

Respond ONLY with a valid JSON object. No markdown, no extra text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_action(text: str) -> SupportAction:
    text = text.strip()
    for fence in ("```json", "```"):
        if fence in text:
            text = text.split(fence, 1)[1].rsplit("```", 1)[0].strip()
            break
    return SupportAction(**json.loads(text))


def action_to_str(action: SupportAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(env: CustomerSupportEnvironment, task_name: str) -> None:
    """
    Runs one full episode and emits exactly:
      1 × [START], N × [STEP], 1 × [END]
    [END] is always emitted, even if an exception occurs.
    """
    step = 0
    rewards: list[float] = []
    success = False
    started = False
    error_already_logged = False  # prevents duplicate [STEP] on re-raise

    try:
        obs: SupportObservation = env.reset(task_name=task_name)

        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)
        started = True

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"New support ticket:\n\n{obs.ticket_content}\n\n"
                    "Please resolve this ticket step by step."
                ),
            },
        ]

        while not obs.done:
            # ── LLM call ──────────────────────────────────────────────────
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.0,
                )
                raw_text = response.choices[0].message.content or ""
                action = parse_action(raw_text)

            except json.JSONDecodeError as exc:
                # Bad JSON — log it, give the model one recovery chance
                step += 1
                rewards.append(0.01)
                err = str(exc).replace("\n", " ")
                print(
                    f"[STEP] step={step} action=null reward=0.01"
                    f" done=false error={err}",
                    flush=True,
                )
                messages.append({"role": "assistant", "content": str(exc)})
                messages.append({
                    "role": "user",
                    "content": "Your last response was not valid JSON. Respond with ONLY a JSON object.",
                })
                if step >= obs.max_steps:
                    break
                continue

            except Exception as exc:
                # API / network error — log and abort (no retry)
                step += 1
                rewards.append(0.01)
                err = str(exc).replace("\n", " ")
                print(
                    f"[STEP] step={step} action=null reward=0.01"
                    f" done=true error={err}",
                    flush=True,
                )
                error_already_logged = True
                raise

            # ── Environment step ──────────────────────────────────────────
            obs = env.step(action)
            step += 1
            rewards.append(obs.reward)

            error_str = obs.error_message if obs.error_message else "null"
            print(
                f"[STEP] step={step} action={action_to_str(action)}"
                f" reward={obs.reward:.2f} done={str(obs.done).lower()}"
                f" error={error_str}",
                flush=True,
            )

            if not obs.done:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({
                    "role": "user",
                    "content": obs.tool_result or "Action processed. Continue.",
                })

        success = obs.success

    except Exception as exc:
        if not error_already_logged:
            err = str(exc).replace("\n", " ")
            step += 1
            rewards.append(0.01)
            print(
                f"[STEP] step={step} action=null reward=0.01"
                f" done=true error={err}",
                flush=True,
            )

    finally:
        # close() the env if the interface supports it
        try:
            env.close()
        except AttributeError:
            pass

        if started:
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(
                f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    env = CustomerSupportEnvironment()
    for task_name in TASKS:
        run_task(env, task_name)


if __name__ == "__main__":
    main()
