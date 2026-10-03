"""Simulator-independent discrete visual navigation episode execution."""

from __future__ import annotations

import base64
import json
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    """One first-person camera observation; the hidden goal is never included."""

    jpeg: bytes
    depth_f32: bytes | None = None
    depth_width: int | None = None
    depth_height: int | None = None

    def __post_init__(self) -> None:
        fields = (self.depth_f32, self.depth_width, self.depth_height)
        if all(value is None for value in fields):
            return
        if any(value is None for value in fields):
            raise ValueError("depth data and dimensions must be supplied together")
        if self.depth_width < 1 or self.depth_height < 1:
            raise ValueError("depth dimensions must be positive")
        if len(self.depth_f32) != self.depth_width * self.depth_height * 4:
            raise ValueError("depth data must contain one float32 value per pixel")


@dataclass(frozen=True)
class EpisodeRequest:
    """Common instruction and opaque task-specific episode payload."""

    episode_id: str
    instruction: str
    payload: object
    allowed_actions: frozenset[str]
    terminal_action: str


class EpisodeSimulator(ABC):
    """Concrete simulator implementation used by the common episode runner."""

    @abstractmethod
    def reset(self, episode: EpisodeRequest) -> list[float]:
        """Reset scene and return initial position in the episode's coordinates."""

    @abstractmethod
    def observe(self) -> Observation:
        """Render the current agent view."""

    @abstractmethod
    def apply(self, action: str) -> list[float]:
        """Execute one action and return its resulting position."""

    @abstractmethod
    def close(self) -> None:
        """Release simulator resources."""


class Policy(ABC):
    """Action policy with an explicit per-episode history reset."""

    @abstractmethod
    def reset(self, episode_id: str) -> None:
        """Clear all history for the next episode."""

    @abstractmethod
    def act(
        self, episode_id: str, frame_id: int, instruction: str, observation: Observation
    ) -> str:
        """Return a valid action for this exact observation."""


class HttpPolicy(Policy):
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _request(self, route: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.url + route,
            json.dumps(payload).encode("utf-8"),
            {"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.load(response)

    def reset(self, episode_id: str) -> None:
        response = self._request("/reset", {"episode_id": episode_id})
        if response.get("episode_id") != episode_id:
            raise ValueError("policy reset did not acknowledge episode ID")

    def act(
        self, episode_id: str, frame_id: int, instruction: str, observation: Observation
    ) -> str:
        payload = {
            "episode_id": episode_id,
            "frame_id": frame_id,
            "instruction": instruction,
            "jpeg_base64": base64.b64encode(observation.jpeg).decode("ascii"),
        }
        if observation.depth_f32 is not None:
            payload.update(
                {
                    "depth_f32_base64": base64.b64encode(observation.depth_f32).decode("ascii"),
                    "depth_width": observation.depth_width,
                    "depth_height": observation.depth_height,
                }
            )
        response = self._request("/act", payload)
        if response.get("episode_id") != episode_id or response.get("frame_id") != frame_id:
            raise ValueError("stale policy action")
        action = response.get("action")
        if not isinstance(action, str) or not action:
            raise ValueError(f"invalid action {action!r}")
        return action


def run_episode(
    simulator: EpisodeSimulator, policy: Policy, episode: EpisodeRequest, max_steps: int
) -> dict:
    """Run a paired episode and return raw measurements for task scoring."""
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if episode.terminal_action not in episode.allowed_actions:
        raise ValueError("terminal action must be allowed")
    episode_id = episode.episode_id
    policy.reset(episode_id)
    positions = [simulator.reset(episode)]
    actions = []
    stopped = False
    instruction = episode.instruction
    for frame_id in range(max_steps):
        observation = simulator.observe()
        action = policy.act(episode_id, frame_id, instruction, observation)
        if action not in episode.allowed_actions:
            raise ValueError(f"invalid action {action!r}")
        actions.append(action)
        if action == episode.terminal_action:
            stopped = True
            break
        positions.append(simulator.apply(action))
    return {"positions": positions, "actions": actions, "stopped": stopped}
