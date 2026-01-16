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

## Prerequisites
```bash
# Install libnuma (required for NUMA affinity in training)
sudo apt-get install -y libnuma-dev

# Install dependencies from skyrl-train directory
cd skyrl-train
uv sync --extra vllm
```

## How to run (shaped rewards)
1) Generate data (must use module syntax from skyrl-train directory):
```bash
cd skyrl-train
uv run python -m examples.catch.catch_dataset \
  --output_dir "$HOME/data/catch" \
  --grid_w 7 --grid_h 6 --max_turns 5 \
  --reward_mode shaped
```

2) Train (requires batched mode and logger config):
```bash
cd skyrl-train
bash examples/catch/run_catch.sh generator.batched=true trainer.logger=tensorboard
```

**Note:** The script has `generator.batched=false` but vLLM offline engine requires batched mode. Override with `generator.batched=true`.

**Note:** Default logger is wandb which requires WANDB_API_KEY. Use `trainer.logger=tensorboard` if you don't have one.

## Switching to episodic
Regenerate data with `--reward_mode episodic`, then re-run training.

## Known issues

### Memory issues with Qwen3-1.7B on single GPU
The 1.7B model causes memory issues on single 4090 (24GB VRAM, ~38GB RAM).

**Root cause:** SkyRL originally loaded models in fp32 for FSDP numerical stability. This has been **fixed** - models now load in bf16 when `trainer.bf16=true` (default).

**Changes made to `skyrl_train/workers/fsdp/fsdp_worker.py`:**
- Policy model (line ~129): Changed `bf16=False` to `bf16=self.cfg.trainer.bf16`
- Critic model (line ~291): Same change

**Current status with 1.7B:**
- bf16 model loading works (halves memory usage)
- vLLM needs more GPU memory for KV cache than available after training models load
- Need to balance `gpu_memory_utilization` with training model memory requirements

**Working configuration (Qwen3-0.6B):**
```bash
bash examples/catch/run_catch.sh \
  generator.batched=true \
  trainer.logger=tensorboard \
  trainer.policy.model.path=Qwen/Qwen3-0.6B-Base \
  trainer.train_batch_size=64 \
  trainer.policy_mini_batch_size=16 \
  trainer.critic_mini_batch_size=16 \
  trainer.eval_batch_size=64 \
  generator.gpu_memory_utilization=0.4
```

**For 1.7B to work on single GPU:**
- May need to disable ref model or use LoRA to reduce memory
- Or use multiple GPUs with model sharding

## Next steps / checkpoints
- Fix OOM and verify training runs to completion.
- Verify shaped reward learns (track `catch_rate` in env metrics).
- Tune batch sizes and vLLM settings for single GPU stability.
- Switch to episodic reward and compare learning curves.

## Keep this file up to date
If you change the Catch setup, reward design, run commands, or defaults, update this file immediately so the next agent can pick up without re-discovery.
