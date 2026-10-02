"""R2R-specific policy transport, separate from the simulator-neutral runner."""

from __future__ import annotations

from bench.episode import HttpPolicy

from isaac_r2r.cma_inputs import validate_instruction_tokens


class R2RHttpPolicy(HttpPolicy):
    """Add audited CMA token IDs to each action request when supplied."""

    def __init__(
        self, url: str, instruction_tokens: list[int] | None = None, policy_id: str | None = None
    ):
        super().__init__(url)
        if instruction_tokens is not None:
            validate_instruction_tokens(instruction_tokens)
        self.instruction_tokens = instruction_tokens
        self.policy_id = policy_id

    def reset(self, episode_id: str) -> None:
        if self.policy_id is None:
            return super().reset(episode_id)
        response = self._request("/reset", {"episode_id": episode_id})
        if response.get("episode_id") != episode_id:
            raise ValueError("policy reset did not acknowledge episode ID")
        if response.get("policy_id") != self.policy_id:
            raise ValueError("policy server identity differs from --policy-id")

    def _request(self, route: str, payload: dict) -> dict:
        if route == "/act" and self.instruction_tokens is not None:
            payload = {**payload, "instruction_tokens": self.instruction_tokens}
        return super()._request(route, payload)
