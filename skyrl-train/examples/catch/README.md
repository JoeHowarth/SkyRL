### Catch (text-based, single-token actions)

This example trains a small model to play a simple Catch game with **single-token actions** (L/R/S). It starts with **shaped rewards** and can be switched to episodic rewards by modifying the dataset `reward_spec`. The prompt template is defined in `examples/catch/prompt.py`.

#### 1) Generate the dataset

```bash
uv run examples/catch/catch_dataset.py \
  --output_dir "$HOME/data/catch" \
  --grid_w 7 --grid_h 6 --max_turns 5 \
  --reward_mode shaped
```

#### 2) Launch training (single GPU + vLLM)

```bash
bash examples/catch/run_catch.sh
```

#### Switch to episodic rewards

Regenerate the dataset with:

```bash
uv run examples/catch/catch_dataset.py \
  --output_dir "$HOME/data/catch" \
  --grid_w 7 --grid_h 6 --max_turns 5 \
  --reward_mode episodic
```
