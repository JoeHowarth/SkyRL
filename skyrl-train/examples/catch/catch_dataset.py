"""
Preprocess a dataset for the Catch environment in parquet format.
"""

import argparse
import os
import random
from datasets import Dataset

from examples.catch.prompt import SYSTEM_PROMPT, format_state


def create_example(
    rng: random.Random,
    grid_w: int,
    grid_h: int,
    max_turns: int,
    split_name: str,
    reward_mode: str,
    step_penalty: float,
    terminal_catch: float,
    terminal_miss: float,
):
    ball_x = rng.randrange(grid_w)
    ball_y = 0
    paddle_x = rng.randrange(grid_w)

    system_prompt = {"role": "system", "content": SYSTEM_PROMPT}

    state_text = format_state(
        grid_w=grid_w,
        grid_h=grid_h,
        ball_x=ball_x,
        ball_y=ball_y,
        paddle_x=paddle_x,
        turns_left=max_turns,
    )

    data = {
        "data_source": "synthetic_catch",
        "prompt": [system_prompt, {"role": "user", "content": state_text}],
        "env_class": "catch",
        "reward_spec": {
            "mode": reward_mode,
            "step_penalty": step_penalty,
            "terminal_catch": terminal_catch,
            "terminal_miss": terminal_miss,
        },
        "extra_info": {
            "grid_w": grid_w,
            "grid_h": grid_h,
            "max_turns": max_turns,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "paddle_x": paddle_x,
            "split": split_name,
        },
    }
    return data


def create_dataset(
    num_examples: int,
    grid_w: int,
    grid_h: int,
    max_turns: int,
    split_name: str,
    seed: int,
    reward_mode: str,
    step_penalty: float,
    terminal_catch: float,
    terminal_miss: float,
):
    rng = random.Random(seed)
    examples = [
        create_example(
            rng,
            grid_w,
            grid_h,
            max_turns,
            split_name,
            reward_mode,
            step_penalty,
            terminal_catch,
            terminal_miss,
        )
        for _ in range(num_examples)
    ]
    return Dataset.from_list(examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/catch")
    parser.add_argument("--grid_w", type=int, default=7)
    parser.add_argument("--grid_h", type=int, default=6)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=20000)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--reward_mode", type=str, default="shaped", choices=["shaped", "episodic"])
    parser.add_argument("--step_penalty", type=float, default=-0.01)
    parser.add_argument("--terminal_catch", type=float, default=1.0)
    parser.add_argument("--terminal_miss", type=float, default=-1.0)

    args = parser.parse_args()

    train_dataset = create_dataset(
        args.train_size,
        args.grid_w,
        args.grid_h,
        args.max_turns,
        "train",
        seed=args.seed,
        reward_mode=args.reward_mode,
        step_penalty=args.step_penalty,
        terminal_catch=args.terminal_catch,
        terminal_miss=args.terminal_miss,
    )
    val_dataset = create_dataset(
        args.test_size,
        args.grid_w,
        args.grid_h,
        args.max_turns,
        "test",
        seed=args.seed + 1,
        reward_mode=args.reward_mode,
        step_penalty=args.step_penalty,
        terminal_catch=args.terminal_catch,
        terminal_miss=args.terminal_miss,
    )

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "validation.parquet"))

    print(f"Generated {args.train_size} training examples and {args.test_size} test examples")
    print(f"Grid: {args.grid_w}x{args.grid_h}, max_turns={args.max_turns}")
    print(f"Saved to {output_dir}")
