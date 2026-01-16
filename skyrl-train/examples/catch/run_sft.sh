#!/bin/bash
set -ex

# Configuration
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-0.6B-Base}"
DATA_DIR="${DATA_DIR:-$HOME/data/catch_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/models/catch_sft}"
ACTION_MODE="${ACTION_MODE:-random}"  # "random" to learn format only, "optimal" to learn policy too

# Step 1: Generate SFT dataset
echo "=== Generating SFT dataset (action_mode=$ACTION_MODE) ==="
uv run python -m examples.catch.sft_dataset \
    --output_dir "$DATA_DIR" \
    --action_mode "$ACTION_MODE" \
    --train_size 5000 \
    --test_size 500

# Step 2: Run SFT training
echo "=== Running SFT training ==="
uv run python -m examples.catch.run_sft \
    --model_name "$MODEL_NAME" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5

echo "=== SFT complete! Model saved to $OUTPUT_DIR ==="
echo ""
echo "Now run RL with the SFT model:"
echo "  bash examples/catch/run_catch.sh \\"
echo "    trainer.policy.model.path=$OUTPUT_DIR \\"
echo "    generator.batched=true \\"
echo "    trainer.logger=tensorboard"
