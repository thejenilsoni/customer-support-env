"""
FastAPI application for the Customer Support Environment.

Routes:
  /              → serves static/index.html (dashboard UI)
  /static/*      → static assets
  /api/run       → POST  streaming inference endpoint (used by UI)
  /reset         → POST  OpenEnv step
  /step          → POST  OpenEnv step
  /state         → GET   OpenEnv state
  /health        → GET   health check
  /docs          → GET   Swagger UI
"""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:
    raise ImportError(
        "openenv-core is not installed. Run: pip install 'openenv-core[core]>=0.2.2'"
    ) from exc

try:
    from models import SupportAction, SupportObservation
except ImportError:
    from ..models import SupportAction, SupportObservation

from .customer_support_environment import CustomerSupportEnvironment
from openai import OpenAI

# ---------------------------------------------------------------------------
# OpenEnv FastAPI app
# ---------------------------------------------------------------------------

app = create_app(
    CustomerSupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="customer_support",
    max_concurrent_envs=1,
)

# ---------------------------------------------------------------------------
# Static files  (dashboard UI)
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).parent.parent / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# /api/run  — streaming inference endpoint
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert customer support agent. Resolve support tickets efficiently.

Available tools — respond with a single JSON object:

{"action_type": "lookup_knowledge_base", "query": "<terms>"}
{"action_type": "lookup_order",          "order_id": "ORD-XXXX"}
{"action_type": "lookup_account",        "account_id": "ACC-XXXX"}
{"action_type": "lookup_billing",        "account_id": "ACC-XXXX"}
{"action_type": "submit_resolution",     "resolution_type": "refund|credit|account_fix|explanation|escalate", "message": "<reply>"}
{"action_type": "escalate",              "message": "<reason>"}

Strategy: gather information first, check KB for policies, then submit a complete resolution.
Never repeat the same lookup. Respond ONLY with a valid JSON object."""

HF_ROUTER_BASE = "https://router.huggingface.co/v1"
BENCHMARK = "customer_support_env"


class RunRequest(BaseModel):
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    api_base_url: str = HF_ROUTER_BASE
    tasks: list[str] = ["easy_refund", "medium_account_lockout", "hard_billing_dispute"]


def _parse_action(text: str) -> SupportAction:
    text = text.strip()
    for fence in ("```json", "```"):
        if fence in text:
            text = text.split(fence, 1)[1].rsplit("```", 1)[0].strip()
            break
    return SupportAction(**json.loads(text))


def _action_to_str(action: SupportAction) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))


def _run_all_tasks_sync(req: RunRequest, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    """Runs in a background thread. Puts log lines into the async queue."""

    def emit(line: str) -> None:
        asyncio.run_coroutine_threadsafe(queue.put(line), loop).result()

    token = os.getenv("HF_TOKEN") or os.getenv("HF_TOKEN_SECRET") or ""
    client = OpenAI(base_url=req.api_base_url, api_key=token)
    env = CustomerSupportEnvironment()

    for task_name in req.tasks:
        step = 0
        rewards: list[float] = []
        success = False
        started = False
        error_logged = False

        try:
            obs = env.reset(task_name=task_name)
            emit(f"[START] task={task_name} env={BENCHMARK} model={req.model_name}")
            started = True

            messages: list[dict] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"New support ticket:\n\n{obs.ticket_content}\n\nPlease resolve this ticket."},
            ]

            while not obs.done:
                try:
                    response = client.chat.completions.create(
                        model=req.model_name,
                        messages=messages,
                        temperature=0.0,
                    )
                    raw = response.choices[0].message.content or ""
                    action = _parse_action(raw)

                except json.JSONDecodeError as exc:
                    step += 1
                    rewards.append(0.0)
                    err = str(exc).replace("\n", " ")
                    emit(f"[STEP] step={step} action=null reward=0.00 done=false error={err}")
                    messages.append({"role": "assistant", "content": str(exc)})
                    messages.append({"role": "user", "content": "Respond with ONLY a valid JSON object."})
                    if step >= obs.max_steps:
                        break
                    continue

                except Exception as exc:
                    step += 1
                    rewards.append(0.0)
                    err = str(exc).replace("\n", " ")
                    emit(f"[STEP] step={step} action=null reward=0.00 done=true error={err}")
                    error_logged = True
                    raise

                obs = env.step(action)
                step += 1
                rewards.append(obs.reward)
                err_str = obs.error_message if obs.error_message else "null"
                emit(
                    f"[STEP] step={step} action={_action_to_str(action)}"
                    f" reward={obs.reward:.2f} done={str(obs.done).lower()} error={err_str}"
                )

                if not obs.done:
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": obs.tool_result or "Action processed. Continue."})

            success = obs.success

        except Exception as exc:
            if not error_logged:
                step += 1
                rewards.append(0.0)
                err = str(exc).replace("\n", " ")
                emit(f"[STEP] step={step} action=null reward=0.00 done=true error={err}")
        finally:
            try:
                env.close()
            except AttributeError:
                pass
            if started:
                rewards_str = ",".join(f"{r:.2f}" for r in rewards)
                emit(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")

    asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()  # sentinel


@app.post("/api/run")
async def api_run(req: RunRequest):
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    thread = threading.Thread(
        target=_run_all_tasks_sync,
        args=(req, queue, loop),
        daemon=True,
    )
    thread.start()

    async def generate():
        while True:
            line = await queue.get()
            if line is None:
                break
            yield (line + "\n").encode()

    return StreamingResponse(generate(), media_type="text/plain")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
