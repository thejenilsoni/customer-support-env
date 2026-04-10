"""
HTTP client for the Customer Support Environment.

Usage (async):
    async with CustomerSupportClient(base_url="http://localhost:7860") as env:
        obs = await env.reset(task_name="easy_refund")
        obs = await env.step(SupportAction(action_type="lookup_order", order_id="ORD-1001"))

Usage (sync):
    with CustomerSupportClient(base_url="http://localhost:7860").sync() as env:
        obs = env.reset(task_name="easy_refund")
"""

from typing import Optional

try:
    from openenv.core.env_client.client import EnvClient
except ImportError:
    from openenv.core.env_client.client import EnvClient

from .models import SupportAction, SupportObservation


class CustomerSupportClient(EnvClient[SupportAction, SupportObservation]):
    """
    Async client for the CustomerSupportEnvironment server.
    Wraps the standard OpenEnv HTTP/WebSocket transport.
    """

    def __init__(self, base_url: str = "http://localhost:7860", **kwargs):
        super().__init__(
            base_url=base_url,
            action_type=SupportAction,
            observation_type=SupportObservation,
            **kwargs,
        )

    async def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs,
    ) -> SupportObservation:
        return await super().reset(
            seed=seed,
            episode_id=episode_id,
            task_name=task_name,
            **kwargs,
        )


__all__ = ["CustomerSupportClient"]
