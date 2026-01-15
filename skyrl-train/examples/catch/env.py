from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import random

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

from examples.catch.prompt import format_state


@dataclass
class CatchState:
    ball_x: int
    ball_y: int
    paddle_x: int


class CatchEnv(BaseTextEnv):
    """
    Simple text-based Catch environment.

    - Grid width W, height H
    - Ball starts at y=0 and falls +1 each step
    - Paddle is on bottom row (y=H-1)
    - Actions: L (left), R (right), S (stay)
    """

    def __init__(self, env_config: Dict[str, Any] = {}, extras: Dict[str, Any] = {}):
        super().__init__()

        reward_spec = extras.get("reward_spec", {})
        extra_info = extras.get("extra_info", {})

        self.grid_w = int(extra_info.get("grid_w", 7))
        self.grid_h = int(extra_info.get("grid_h", 6))
        self.max_turns = int(extra_info.get("max_turns", self.grid_h - 1))

        seed = extra_info.get("seed", None)
        self._rng = random.Random(seed)

        # Initial state can be provided, otherwise randomize
        ball_x = extra_info.get("ball_x", self._rng.randrange(self.grid_w))
        ball_y = extra_info.get("ball_y", 0)
        paddle_x = extra_info.get("paddle_x", self._rng.randrange(self.grid_w))
        self.state = CatchState(ball_x=ball_x, ball_y=ball_y, paddle_x=paddle_x)

        # Reward settings
        self.reward_mode = reward_spec.get("mode", "shaped")  # "shaped" or "episodic"
        self.step_penalty = float(reward_spec.get("step_penalty", -0.01))
        self.terminal_catch = float(reward_spec.get("terminal_catch", 1.0))
        self.terminal_miss = float(reward_spec.get("terminal_miss", -1.0))

    def init(self, prompt):
        # No special pre-processing; return prompt and empty metadata
        return prompt, {}

    def _parse_action(self, action: str) -> str:
        if not action:
            return "S"
        stripped = action.strip()
        if not stripped:
            return "S"
        token = stripped[0].upper()
        if token not in {"L", "R", "S"}:
            return "S"
        return token

    def _apply_action(self, token: str) -> None:
        if token == "L":
            self.state.paddle_x = max(0, self.state.paddle_x - 1)
        elif token == "R":
            self.state.paddle_x = min(self.grid_w - 1, self.state.paddle_x + 1)
        # "S" => no change

    def _phi(self, state: CatchState) -> float:
        # Potential based on horizontal distance; range [-1, 0]
        max_dist = max(1, self.grid_w - 1)
        return -abs(state.ball_x - state.paddle_x) / max_dist

    def _is_terminal(self) -> bool:
        return self.state.ball_y >= self.grid_h - 1 or self.turns >= self.max_turns

    def _caught(self) -> bool:
        return self.state.ball_y >= self.grid_h - 1 and self.state.ball_x == self.state.paddle_x

    def _state_text(self) -> str:
        return format_state(
            grid_w=self.grid_w,
            grid_h=self.grid_h,
            ball_x=self.state.ball_x,
            ball_y=self.state.ball_y,
            paddle_x=self.state.paddle_x,
            turns_left=max(0, self.max_turns - self.turns),
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        prev_state = CatchState(**vars(self.state))

        token = self._parse_action(action)
        self._apply_action(token)

        # Ball falls one step
        self.state.ball_y += 1

        done = self._is_terminal()
        caught = self._caught()

        reward = 0.0
        if self.reward_mode == "shaped":
            reward += (self._phi(self.state) - self._phi(prev_state)) + self.step_penalty
        if done:
            reward += self.terminal_catch if caught else self.terminal_miss

        observations = [] if done else [{"role": "user", "content": self._state_text()}]

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=done,
            metadata={"caught": caught, "action": token},
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "steps": self.turns,
            "caught": int(self._caught()),
        }

    @staticmethod
    def aggregate_metrics(metrics: list[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        n = len(metrics)
        avg_steps = sum(float(m.get("steps", 0)) for m in metrics) / n
        catch_rate = sum(float(m.get("caught", 0)) for m in metrics) / n
        return {"avg_steps": avg_steps, "catch_rate": catch_rate}
