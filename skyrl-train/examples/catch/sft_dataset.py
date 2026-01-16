"""
Generate SFT dataset to teach the model to output L/R/S tokens.
Uses random actions so RL can learn the actual policy.
"""

import argparse
import os
import random
from datasets import Dataset

from examples.catch.prompt import SYSTEM_PROMPT, format_state


def compute_optimal_action(ball_x: int, paddle_x: int) -> str:
    """Compute the optimal action to move paddle towards ball."""
    if paddle_x < ball_x:
        return "R"
    elif paddle_x > ball_x:
        return "L"
    else:
        return "S"


def create_sft_example(
    rng: random.Random,
    grid_w: int,
    grid_h: int,
    max_turns: int,
    action_mode: str,  # "random" or "optimal"
):
    ball_x = rng.randrange(grid_w)
    ball_y = rng.randrange(grid_h - 1)  # Ball can be at various heights
    paddle_x = rng.randrange(grid_w)
    turns_left = rng.randint(1, max_turns)

    state_text = format_state(
        grid_w=grid_w,
        grid_h=grid_h,
        ball_x=ball_x,
        ball_y=ball_y,
        paddle_x=paddle_x,
        turns_left=turns_left,
    )

    # Choose action based on mode
    if action_mode == "optimal":
        action = compute_optimal_action(ball_x, paddle_x)
    else:  # random
        action = rng.choice(["L", "R", "S"])

    # Format for SFT: messages with assistant response
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state_text},
        {"role": "assistant", "content": action},
    ]

    return {"messages": messages}


def create_sft_dataset(
    num_examples: int,
    grid_w: int,
    grid_h: int,
    max_turns: int,
    seed: int,
    action_mode: str,
):
    rng = random.Random(seed)
    examples = [
        create_sft_example(rng, grid_w, grid_h, max_turns, action_mode)
        for _ in range(num_examples)
    ]
    return Dataset.from_list(examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="~/data/catch_sft")
    parser.add_argument("--grid_w", type=int, default=7)
    parser.add_argument("--grid_h", type=int, default=6)
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--action_mode",
        type=str,
        default="random",
        choices=["random", "optimal"],
        help="random: teach format only, optimal: teach format + policy",
    )

    args = parser.parse_args()

    train_dataset = create_sft_dataset(
        args.train_size,
        args.grid_w,
        args.grid_h,
        args.max_turns,
        seed=args.seed,
        action_mode=args.action_mode,
    )
    val_dataset = create_sft_dataset(
        args.test_size,
        args.grid_w,
        args.grid_h,
        args.max_turns,
        seed=args.seed + 1,
        action_mode=args.action_mode,
    )

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    train_dataset.to_json(os.path.join(output_dir, "train.jsonl"))
    val_dataset.to_json(os.path.join(output_dir, "val.jsonl"))

    print(f"Generated {args.train_size} training examples and {args.test_size} test examples")
    print(f"Action mode: {args.action_mode}")
    print(f"Saved to {output_dir}")
