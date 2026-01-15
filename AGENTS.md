# SkyRL Catch Pilot (Single-GPU)

## What we are doing
We are validating the SkyRL training pipeline by training a small base model (Qwen3 1.7B) to play a simple, fully text-based multi-turn game (Catch) with single-token actions. The plan is to start with shaped rewards, confirm learning signal and stability, then switch to episodic-only rewards.

## Current state
- A Catch example has been scaffolded under `skyrl-train/examples/catch`.
- It includes:
  - `env.py`: Catch environment with shaped/episodic reward modes.
  - `catch_dataset.py`: dataset generator with `--reward_mode shaped|episodic`.
  - `prompt.py`: shared prompt template and state formatting.
  - `main_catch.py`: registers env and launches training.
  - `run_catch.sh`: single-GPU vLLM run script.
  - `README.md`: quick instructions.

## How to run (shaped rewards)
1) Generate data:
```
uv run skyrl-train/examples/catch/catch_dataset.py \
  --output_dir "$HOME/data/catch" \
  --grid_w 7 --grid_h 6 --max_turns 5 \
  --reward_mode shaped
```

2) Train:
```
bash skyrl-train/examples/catch/run_catch.sh
```

## Switching to episodic
Regenerate data with `--reward_mode episodic`, then re-run training.

## Next steps / checkpoints
- Verify shaped reward learns (track `catch_rate` in env metrics).
- Tune batch sizes and vLLM settings for single GPU stability.
- Switch to episodic reward and compare learning curves.

## Keep this file up to date
If you change the Catch setup, reward design, run commands, or defaults, update this file immediately so the next agent can pick up without re-discovery.
