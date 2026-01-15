set -x

# Colocated GRPO training+generation for Qwen3-1.7B on a simple Catch environment.
# uv run examples/catch/catch_dataset.py --output_dir $HOME/data/catch
# export WANDB_API_KEY=<your_key_here>
# bash examples/catch/run_catch.sh

DATA_DIR="$HOME/data/catch"
NUM_GPUS=1

uv run --isolated --extra vllm -m examples.catch.main_catch \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-1.7B-Base" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.critic_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=10 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=128 \
  trainer.policy_mini_batch_size=32 \
  trainer.critic_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.eval_batch_size=128 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=256 \
  generator.sampling_params.max_generate_length=1 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=false \
  generator.batched=false \
  environment.env_class=catch \
  generator.n_samples_per_prompt=4 \
  generator.max_turns=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="wandb" \
  trainer.project_name="catch" \
  trainer.run_name="catch_shaped" \
  $@
