---
title: Customer Support Env
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - agentic-ai
---

# Customer Support Environment

An OpenEnv-compliant reinforcement learning environment that trains agents to
resolve realistic customer support tickets end-to-end.

Agents must gather information using tool calls (knowledge base lookups, order
lookups, account lookups, billing history), then submit a correct and complete
resolution — exactly as a real support agent would.

---

## Environment Overview and Motivation

Customer support is a high-volume, real-world task that requires:
- Multi-step information retrieval (tool use)
- Policy understanding (knowledge base reasoning)
- Structured decision-making (correct resolution type)
- Clear communication (customer-facing message quality)

This environment benchmarks an agent's ability to handle all of these
dimensions across three difficulty levels, providing dense incremental rewards
throughout the episode rather than only at termination.

---

## Action Space

| `action_type`           | Required fields                             | Description                          |
|-------------------------|---------------------------------------------|--------------------------------------|
| `lookup_knowledge_base` | `query: str`                                | Search internal policy articles      |
| `lookup_order`          | `order_id: str`                             | Retrieve order details by ID         |
| `lookup_account`        | `account_id: str`                           | Retrieve account status by ID        |
| `lookup_billing`        | `account_id: str`                           | Retrieve billing history for account |
| `submit_resolution`     | `resolution_type: str`, `message: str`      | Submit final resolution (ends episode) |
| `escalate`              | `message: str`                              | Escalate to senior support (ends episode) |

### `resolution_type` values
`refund` · `credit` · `account_fix` · `explanation` · `escalate`

---

## Observation Space

| Field            | Type           | Description                                      |
|------------------|----------------|--------------------------------------------------|
| `ticket_id`      | `str`          | Unique ticket identifier                         |
| `ticket_content` | `str`          | The customer's original support ticket text      |
| `tool_result`    | `str \| null`  | Output from the last tool call                   |
| `reward`         | `float`        | Reward for the last action (incremental)         |
| `done`           | `bool`         | True when the episode has ended                  |
| `success`        | `bool`         | True when resolved correctly (score ≥ 0.60)      |
| `step_count`     | `int`          | Steps taken so far                               |
| `max_steps`      | `int`          | Maximum allowed steps for this task              |
| `task_name`      | `str`          | Name of the current task                         |
| `task_difficulty`| `str`          | `easy` / `medium` / `hard`                       |

---

## Tasks

### Task 1 — `easy_refund` (Easy)
**Ticket ID:** TKT-0001  
**Scenario:** Customer ordered Wireless Headphones Pro (#ORD-1001). Package
confirmed lost in transit after 25 days. Customer requests a full refund of
$45.99.

**Optimal path (3 steps):**
1. `lookup_order(ORD-1001)` → verify lost status and refund eligibility
2. *(optional)* `lookup_knowledge_base("refund policy")`
3. `submit_resolution(refund, message mentioning amount + timeline)`

**Grading breakdown:**
| Criterion | Weight |
|---|---|
| Looked up order ORD-1001 | 0.35 |
| resolution_type = "refund" | 0.35 |
| Message mentions refund amount ($45.99) | 0.15 |
| Message includes apology or processing timeline | 0.15 |

**Max steps:** 6 | **Baseline score:** ~0.85

---

### Task 2 — `medium_account_lockout` (Medium)
**Ticket ID:** TKT-0002  
**Scenario:** Customer's account (ACC-2047) was auto-locked after a suspicious
login from an unknown IP. They suspect unauthorized access and need help
regaining access.

**Optimal path (4 steps):**
1. `lookup_account(ACC-2047)` → confirm lock reason (suspicious_login)
2. `lookup_knowledge_base("account security")` → get policy
3. `submit_resolution(account_fix, message addressing security + next steps)`
   — or — `escalate` (both acceptable)

**Grading breakdown:**
| Criterion | Weight |
|---|---|
| Looked up account ACC-2047 | 0.25 |
| Checked security policy in KB | 0.20 |
| Correct resolution type (account_fix or escalate) | 0.25 |
| Message mentions security/suspicious activity | 0.15 |
| Message mentions verification steps or timeline | 0.15 |

**Max steps:** 8 | **Baseline score:** ~0.70

---

### Task 3 — `hard_billing_dispute` (Hard)
**Ticket ID:** TKT-0003  
**Scenario:** Customer (ACC-5091) has three concurrent billing issues:
1. Charged $89.99 on March 15 despite cancelling subscription (#ORD-3055) on
   March 10 — **invalid charge, full refund owed**
2. Unrecognised $12.50 charge from March 1 — **valid** (premium API usage)
3. Missing $20 loyalty credit promised in February — **needs to be applied**

**Optimal path (5–7 steps):**
1. `lookup_order(ORD-3055)` → confirm cancellation date
2. `lookup_billing(ACC-5091)` → see all charges, identify valid/invalid
3. `lookup_account(ACC-5091)` → find pending $20 credit
4. *(optional)* KB lookups for billing/credits policy
5. `submit_resolution` addressing all three issues in one message

**Grading breakdown:**
| Criterion | Weight |
|---|---|
| Looked up order ORD-3055 | 0.12 |
| Looked up billing ACC-5091 | 0.12 |
| Looked up account ACC-5091 | 0.11 |
| Refunds invalid $89.99 subscription charge | 0.20 |
| Explains $12.50 as valid API usage charge | 0.15 |
| Addresses missing $20 credit | 0.20 |
| All three issues addressed in one response | 0.10 |

**Max steps:** 12 | **Baseline score:** ~0.50

---

## Reward Function

Rewards are **dense** — issued at every step to guide the agent throughout
the episode, not only at completion.

| Event | Reward |
|---|---|
| Correct, relevant lookup (first time) | +0.10 to +0.30 |
| Relevant KB lookup (first time) | +0.05 to +0.20 |
| Lookup on wrong/irrelevant ID | −0.05 |
| Repeated lookup (same key) | −0.10 |
| `submit_resolution` / `escalate` | grader score (0.0 – 1.0) |
| Timeout (exceeding max steps) | −0.10 penalty |

---

## Setup and Usage

### Local development

```bash
# Install dependencies
pip install "openenv-core[core]>=0.2.2" fastapi uvicorn pydantic openai

# Start the environment server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run the baseline inference script
# Create a token at https://huggingface.co/settings/tokens (read access is enough)
export HF_TOKEN=hf_...

# Defaults use HuggingFace Serverless Inference (free, rate-limited):
python inference.py

# Or override model / endpoint:
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
export API_BASE_URL=https://api-inference.huggingface.co/v1  # default, can omit
python inference.py
```

### Docker

```bash
# Build
docker build -t customer-support-env .

# Run server
docker run -p 7860:7860 customer-support-env

# Run inference against the running container
docker run --network host \
  -e HF_TOKEN=<token> \
  -e MODEL_NAME=gpt-4.1-mini \
  customer-support-env python inference.py
```

### Python API (in-process)

```python
from server.customer_support_environment import CustomerSupportEnvironment
from models import SupportAction

env = CustomerSupportEnvironment()

obs = env.reset(task_name="easy_refund")
print(obs.ticket_content)

obs = env.step(SupportAction(action_type="lookup_order", order_id="ORD-1001"))
print(obs.tool_result, "reward:", obs.reward)

obs = env.step(SupportAction(
    action_type="submit_resolution",
    resolution_type="refund",
    message="Hi Sarah, I've processed a full refund of $45.99. You'll see it in 3-5 business days. Sorry for the trouble!"
))
print("Score:", obs.reward, "Success:", obs.success)
```

---

## Baseline Performance Scores

Measured at temperature=0:

| Task | Difficulty | Llama-3.1-8B (HF free) | gpt-4.1-mini |
|---|---|---|---|
| easy_refund | Easy | ~0.70 | ~0.85 |
| medium_account_lockout | Medium | ~0.55 | ~0.70 |
| hard_billing_dispute | Hard | ~0.35 | ~0.50 |

HF Serverless Inference is free but rate-limited (~10 req/min on most models).
For faster runs use a [Dedicated Inference Endpoint](https://huggingface.co/inference-endpoints) and set `API_BASE_URL` to your endpoint URL.

---

## Validation

```bash
openenv validate
```

Checks that `openenv.yaml` is valid, the environment correctly implements
`reset()`, `step()`, and `state`, and models are properly typed.
